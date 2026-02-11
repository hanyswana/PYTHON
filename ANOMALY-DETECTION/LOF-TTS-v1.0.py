#!/usr/bin/env python3
"""
Train PyOD LOF on normal glucose dataset (415..940 nm).

Flow matches IF-TTS-v1.0.py:
- One global 80/20 train-test split
- Heuristic grid search (TRAIN ONLY), no synthetic anomalies
- Train final model on TRAIN only
- Save best model (.joblib) + ESP32 header (.h)
- Save train/test outputs + summary
"""

import os, sys, joblib
import numpy as np
import pandas as pd

from pyod.models.lof import LOF
from sklearn.model_selection import train_test_split, ParameterGrid


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/raw/Lablink_670_training_normal.csv"

OUT_DIR = f"{DIR}/LOF/result/v1.0"
MODEL_DIR = f"{DIR}/LOF/model/v1.0"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5
SPLIT_SEEDS = [42, 7, 13, 21, 99]

# Heuristic weights
W_FAILRATE = 1.0
W_SPREAD = 0.10
W_STABILITY = 0.50

# LOF grid (keep small; LOF is heavier than IF/MD)
PARAM_GRID = {
    "contamination": [0.01, 0.02, 0.05],
    "n_neighbors": [10, 20, 35, 50],
    "leaf_size": [20, 30, 50],
    "metric": ["minkowski"],
    "p": [1, 2],  # 1=manhattan, 2=euclidean
}


def select_spectral_columns(df: pd.DataFrame, start_col="415 nm", end_col="940 nm") -> pd.DataFrame:
    cols = list(df.columns)
    if start_col not in cols or end_col not in cols:
        raise ValueError(
            f"Cannot find spectral range columns. Missing '{start_col}' or '{end_col}'.\n"
            f"Available columns example: {cols[:10]} ... {cols[-10:]}"
        )
    i0 = cols.index(start_col)
    i1 = cols.index(end_col)
    if i1 < i0:
        raise ValueError(f"Column '{end_col}' appears before '{start_col}'. Check your CSV column order.")
    return df[cols[i0 : i1 + 1]].copy()


def export_pyod_lof_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "LOF_MODEL_H"):
    """
    Export LOF parameters for ESP32 inference.

    Practical note:
    LOF inference requires access to the training reference set (or its neighbor structure),
    so this header will store the reference points used by the fitted detector_.

    This can become LARGE. Still, this exports a functional implementation:
      - store reference X_ref (n_ref x d)
      - compute distances from x to all ref, take k nearest
      - compute simplified LOF score (approx; enough for deployment screening)

    If you want a smaller ESP32 model, MD is preferred.
    """
    bundle = joblib.load(joblib_path)
    pyod_model = bundle["model"]
    spectral_cols = bundle.get("spectral_columns", [])

    # PyOD LOF uses sklearn.neighbors.LocalOutlierFactor inside detector_
    if not hasattr(pyod_model, "detector_"):
        raise ValueError("This PyOD LOF object has no detector_. Did you fit() before saving?")

    det = pyod_model.detector_
    # For novelty usage, LOF keeps training data in _fit_X
    X_ref = getattr(det, "_fit_X", None)
    if X_ref is None:
        # fallback: try attribute _X_train (depends on version)
        X_ref = getattr(det, "_X_train", None)
    if X_ref is None:
        raise ValueError("Could not find LOF reference training data (_fit_X) in the fitted model.")

    X_ref = np.asarray(X_ref, dtype=np.float32)
    n_ref, n_feat = X_ref.shape
    k = int(getattr(det, "n_neighbors", 20))

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("// Auto-generated LOF reference set for ESP32 inference (approx)")
    lines.append("// NOTE: This header can be large because LOF needs the reference set.")
    lines.append("")
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#ifdef ARDUINO")
    lines.append("  #include <Arduino.h>")
    lines.append("  #include <pgmspace.h>")
    lines.append("  #ifndef PROGMEM")
    lines.append("    #define PROGMEM")
    lines.append("  #endif")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#define LOF_N_FEATURES {n_feat}")
    lines.append(f"#define LOF_N_REF {n_ref}")
    lines.append(f"#define LOF_K {k}")
    lines.append("")

    ref_flat = X_ref.reshape(-1)
    lines.append(f"static const float LOF_REF[{n_ref*n_feat}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in ref_flat))
    lines.append("};\n")

    lines.append("static inline float lof_read_f32(const float* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  union { uint32_t u; float f; } v;")
    lines.append("  v.u = (uint32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("  return v.f;")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline float lof_dist2(const float* x, int ref_idx) {")
    lines.append("  float s = 0.0f;")
    lines.append("  int base = ref_idx * LOF_N_FEATURES;")
    lines.append("  for (int j=0; j<LOF_N_FEATURES; ++j) {")
    lines.append("    float r = lof_read_f32(LOF_REF, base + j);")
    lines.append("    float d = x[j] - r;")
    lines.append("    s += d*d;")
    lines.append("  }")
    lines.append("  return s;")
    lines.append("}\n")

    lines.append("// Approx score: mean distance to K nearest reference points (not full LOF ratio).")
    lines.append("static inline float lof_score_approx(const float* x) {")
    lines.append("  // keep K smallest dist2")
    lines.append("  float best[LOF_K];")
    lines.append("  for (int i=0; i<LOF_K; ++i) best[i] = 1e30f;")
    lines.append("  for (int i=0; i<LOF_N_REF; ++i) {")
    lines.append("    float d2 = lof_dist2(x, i);")
    lines.append("    // insert into sorted best (simple O(K))")
    lines.append("    int pos = -1;")
    lines.append("    for (int j=0; j<LOF_K; ++j) {")
    lines.append("      if (d2 < best[j]) { pos = j; break; }")
    lines.append("    }")
    lines.append("    if (pos >= 0) {")
    lines.append("      for (int j=LOF_K-1; j>pos; --j) best[j] = best[j-1];")
    lines.append("      best[pos] = d2;")
    lines.append("    }")
    lines.append("  }")
    lines.append("  float m = 0.0f;")
    lines.append("  for (int i=0; i<LOF_K; ++i) m += sqrtf(best[i]);")
    lines.append("  return m / (float)LOF_K;")
    lines.append("}\n")

    lines.append(f"#endif // {header_guard}\n")

    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return header_path


def heuristic_eval_one_split(X_np: np.ndarray, params: dict, seed: int):
    """
    Fit on train normals, evaluate on val normals only.
    Returns: (val_fail_rate, val_score_std)
    """
    X_train, X_val = train_test_split(X_np, test_size=TEST_SIZE, random_state=seed, shuffle=True)

    clf = LOF(
        n_neighbors=params["n_neighbors"],
        contamination=params["contamination"],
        leaf_size=params["leaf_size"],
        metric=params["metric"],
        p=params["p"],
        novelty=True,          # important for scoring new points
    )
    clf.fit(X_train)

    y_val = clf.predict(X_val)               # 0=pass, 1=fail
    val_fail_rate = float(np.mean(y_val == 1))

    val_scores = clf.decision_function(X_val)  # higher => more anomalous (PyOD)
    val_score_std = float(np.std(val_scores))

    return val_fail_rate, val_score_std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(CSV_PATH)

    X = select_spectral_columns(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()
    meta_clean = df.loc[mask_finite, ["Sample", "glucose"]].reset_index(drop=True)

    X_np = X_clean.to_numpy(dtype=np.float32)

    # 2) ONE global train–test split (80/20)
    X_train, X_test, meta_train, meta_test = train_test_split(
        X_np, meta_clean, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"[Split] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3) Heuristic grid search (TRAIN ONLY)
    results = []
    best_obj = float("inf")
    best_params = None

    for params in ParameterGrid(PARAM_GRID):
        split_fail_rates = []
        split_stds = []

        for seed in SPLIT_SEEDS[:N_SPLITS]:
            fr, sd = heuristic_eval_one_split(X_train, params, seed)
            split_fail_rates.append(fr)
            split_stds.append(sd)

        mean_fail = float(np.mean(split_fail_rates))
        std_fail = float(np.std(split_fail_rates))
        mean_sd = float(np.mean(split_stds))

        obj = (
            W_FAILRATE * abs(mean_fail - float(params["contamination"]))
            + W_SPREAD * mean_sd
            + W_STABILITY * std_fail
        )

        results.append({
            **params,
            "mean_val_fail_rate": mean_fail,
            "std_val_fail_rate": std_fail,
            "mean_val_score_std": mean_sd,
            "objective": obj,
        })

        if best_params is None or obj < best_obj:
            best_obj = obj
            best_params = params

    results_df = pd.DataFrame(results).sort_values("objective", ascending=True)
    grid_path = os.path.join(OUT_DIR, "gridsearch_results_heuristic.csv")
    results_df.to_csv(grid_path, index=False)

    print("[GridSearch] Best params:", best_params)

    # 4) Train FINAL model (TRAIN SET ONLY)
    best_model = LOF(
        n_neighbors=best_params["n_neighbors"],
        contamination=best_params["contamination"],
        leaf_size=best_params["leaf_size"],
        metric=best_params["metric"],
        p=best_params["p"],
        novelty=True,
    )
    best_model.fit(X_train)

    # 5) Save model + ESP32 header
    best_model_path = os.path.join(MODEL_DIR, "lof_pyod_model_best.joblib")
    joblib.dump(
        {
            "model": best_model,
            "spectral_columns": list(X_clean.columns),
            "config": {
                **best_params,
                "random_state": RANDOM_STATE,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "grid_search_mode": "heuristic_train_only",
            },
        },
        best_model_path,
    )

    best_header_path = os.path.join(MODEL_DIR, "lof_pyod_model_best.h")
    export_pyod_lof_joblib_to_header(best_model_path, best_header_path, header_guard="LOF_PYOD_MODEL_BEST_H")

    # 6) TRAIN results (80%)
    y_train = best_model.predict(X_train)
    scores_train = best_model.decision_function(X_train)
    train_results = pd.DataFrame({
        "Sample": meta_train["Sample"].values,
        "glucose": meta_train["glucose"].values,
        "is_anomaly": y_train,
        "status": np.where(y_train == 1, "fail", "pass"),
        "anomaly_score": scores_train,
        "split": "train",
    })
    train_results.to_csv(os.path.join(OUT_DIR, "anomaly_results_train.csv"), index=False)

    # 7) TEST results (20%)
    y_test = best_model.predict(X_test)
    scores_test = best_model.decision_function(X_test)
    test_results = pd.DataFrame({
        "Sample": meta_test["Sample"].values,
        "glucose": meta_test["glucose"].values,
        "is_anomaly": y_test,
        "status": np.where(y_test == 1, "fail", "pass"),
        "anomaly_score": scores_test,
        "split": "test",
    })
    test_results.to_csv(os.path.join(OUT_DIR, "anomaly_results_test.csv"), index=False)

    # 8) Summary
    def summarize(y):
        return {
            "total": int(len(y)),
            "fail": int((y == 1).sum()),
            "pass": int((y == 0).sum()),
            "fail_rate_percent": float((y == 1).mean() * 100.0),
        }

    summary = {
        "train": summarize(y_train),
        "test": summarize(y_test),
        "best_params": best_params,
    }

    summary_path = os.path.join(OUT_DIR, "anomaly_summary_best.txt")
    with open(summary_path, "w") as f:
        f.write("LOF (Heuristic, Train/Test Split)\n\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples:  {len(X_test)}\n\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write(f"Train summary: {summary['train']}\n")
        f.write(f"Test summary:  {summary['test']}\n")

    print("Done ✅")
    print("Training results  → anomaly_results_train.csv")
    print("Testing results   → anomaly_results_test.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
