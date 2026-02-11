#!/usr/bin/env python3
"""
Train PyOD One-Class SVM (OCSVM) on normal glucose dataset (415..940 nm).

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

from pyod.models.ocsvm import OCSVM
from sklearn.model_selection import train_test_split, ParameterGrid


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/raw/Lablink_670_training_normal.csv"

OUT_DIR = f"{DIR}/SVM/result/v1.0"
MODEL_DIR = f"{DIR}/SVM/model/v1.0"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5
SPLIT_SEEDS = [42, 7, 13, 21, 99]  # length should match N_SPLITS

# Heuristic weights
W_FAILRATE = 1.0
W_SPREAD = 0.10
W_STABILITY = 0.50

# OCSVM grid
# Notes:
# - In One-Class SVM, `nu` is an upper bound on training outlier fraction (and lower bound on SV fraction).
#   It is the closest analogue to "contamination". We'll treat it that way in the heuristic objective.
# - `gamma` controls RBF kernel width. Too high => very tight boundary (more fails). Too low => too loose.
PARAM_GRID = {
    "nu": [0.01, 0.02, 0.05],
    "kernel": ["rbf"],
    "gamma": ["scale", "auto", 0.1, 0.5, 1.0],
    # keep default degree/coef0 irrelevant for rbf
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
    spectral_cols = cols[i0 : i1 + 1]
    return df[spectral_cols].copy()


def export_pyod_ocsvm_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "OCSVM_MODEL_H"):
    """
    Export trained OCSVM parameters for ESP32 inference.

    For RBF kernel One-Class SVM:
      f(x) = sum_i alpha_i * exp(-gamma * ||x - sv_i||^2) + intercept
    Predict fail if f(x) < 0 (OCSVM decision boundary).

    PyOD's OCSVM wraps sklearn.svm.OneClassSVM in `detector_`.
    """
    bundle = joblib.load(joblib_path)
    pyod_model = bundle["model"]
    spectral_cols = bundle.get("spectral_columns", [])

    if not hasattr(pyod_model, "detector_"):
        raise ValueError("This PyOD OCSVM object has no detector_. Did you fit() before saving?")

    sk = pyod_model.detector_  # sklearn OneClassSVM

    # support vectors and dual coef
    sv = np.asarray(sk.support_vectors_, dtype=np.float32)        # (n_sv, n_feat)
    dual = np.asarray(sk.dual_coef_, dtype=np.float32).reshape(-1)  # (n_sv,) usually
    intercept = float(np.asarray(sk.intercept_, dtype=np.float32).reshape(-1)[0])

    # gamma in sklearn can be 'scale'/'auto' or numeric; after fitting, sklearn stores _gamma
    gamma = float(getattr(sk, "_gamma", np.nan))
    if not np.isfinite(gamma):
        # last resort: try reading attribute 'gamma' if it's numeric
        g = getattr(sk, "gamma", None)
        if isinstance(g, (float, int)):
            gamma = float(g)
        else:
            raise ValueError("Could not determine numeric gamma for fitted OCSVM (sklearn _gamma missing).")

    n_sv, n_feat = sv.shape
    if n_feat != len(spectral_cols):
        # It's okay if you didn't store cols; but if stored, this should match.
        pass

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("// Auto-generated One-Class SVM (RBF) parameters")
    lines.append("// score(x) = sum_i alpha_i * exp(-gamma * ||x - sv_i||^2) + intercept")
    lines.append("// decision: fail if score(x) < 0, pass otherwise")
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
    lines.append(f"#define OCSVM_N_FEATURES {n_feat}")
    lines.append(f"#define OCSVM_N_SV {n_sv}")
    lines.append(f"#define OCSVM_GAMMA {gamma:.10f}f")
    lines.append(f"#define OCSVM_INTERCEPT {intercept:.10f}f")
    lines.append("")

    # Store SVs row-major
    sv_flat = sv.reshape(-1)
    lines.append(f"static const float OCSVM_SV[{n_sv*n_feat}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in sv_flat))
    lines.append("};\n")

    # Dual coefficients (alpha)
    lines.append(f"static const float OCSVM_ALPHA[{n_sv}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in dual))
    lines.append("};\n")

    lines.append("static inline float svm_read_f32(const float* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  union { uint32_t u; float f; } v;")
    lines.append("  v.u = (uint32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("  return v.f;")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline float ocsvm_score(const float* x) {")
    lines.append("  const float gamma = OCSVM_GAMMA;")
    lines.append("  float s = OCSVM_INTERCEPT;")
    lines.append("  for (int i=0; i<OCSVM_N_SV; ++i) {")
    lines.append("    float dist2 = 0.0f;")
    lines.append("    int base = i * OCSVM_N_FEATURES;")
    lines.append("    for (int j=0; j<OCSVM_N_FEATURES; ++j) {")
    lines.append("      float svv = svm_read_f32(OCSVM_SV, base + j);")
    lines.append("      float d = x[j] - svv;")
    lines.append("      dist2 += d * d;")
    lines.append("    }")
    lines.append("    float a = svm_read_f32(OCSVM_ALPHA, i);")
    lines.append("    s += a * expf(-gamma * dist2);")
    lines.append("  }")
    lines.append("  return s;")
    lines.append("}\n")

    lines.append("static inline int ocsvm_predict(const float* x) {")
    lines.append("  // return 0=pass, 1=fail")
    lines.append("  float s = ocsvm_score(x);")
    lines.append("  return (s < 0.0f) ? 1 : 0;")
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

    clf = OCSVM(
        nu=params["nu"],
        kernel=params["kernel"],
        gamma=params["gamma"],
    )
    clf.fit(X_train)

    # PyOD convention:
    # predict(X): 0=inlier(pass), 1=outlier(fail)
    y_val = clf.predict(X_val)
    val_fail_rate = float(np.mean(y_val == 1))

    val_scores = clf.decision_function(X_val)  # higher => more anomalous (PyOD convention)
    val_score_std = float(np.std(val_scores))

    return val_fail_rate, val_score_std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------------------------------------------------
    # 1) Load full dataset
    # -------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # Select spectral features
    X = select_spectral_columns(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()

    meta_clean = df.loc[mask_finite, ["Sample", "glucose"]].reset_index(drop=True)

    X_np = X_clean.to_numpy().astype(np.float32)

    # -------------------------------------------------
    # 2) ONE global train–test split (80/20)
    # -------------------------------------------------
    X_train, X_test, meta_train, meta_test = train_test_split(
        X_np,
        meta_clean,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"[Split] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # -------------------------------------------------
    # 3) Heuristic grid search (TRAIN ONLY)
    # -------------------------------------------------
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

        # here, expected fail-rate is approx nu (heuristic)
        expected = float(params["nu"])
        obj = (
            W_FAILRATE * abs(mean_fail - expected)
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

    # -------------------------------------------------
    # 4) Train FINAL model (TRAIN SET ONLY)
    # -------------------------------------------------
    best_model = OCSVM(
        nu=best_params["nu"],
        kernel=best_params["kernel"],
        gamma=best_params["gamma"],
    )
    best_model.fit(X_train)

    # -------------------------------------------------
    # 5) Save model + ESP32 header
    # -------------------------------------------------
    best_model_path = os.path.join(MODEL_DIR, "ocsvm_pyod_model_best.joblib")
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

    best_header_path = os.path.join(MODEL_DIR, "ocsvm_pyod_model_best.h")
    export_pyod_ocsvm_joblib_to_header(
        best_model_path,
        best_header_path,
        header_guard="OCSVM_PYOD_MODEL_BEST_H",
    )

    # -------------------------------------------------
    # 6) TRAINING results (80%)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 7) TEST results (20%, unseen)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 8) Summary
    # -------------------------------------------------
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
        f.write("One-Class SVM (Heuristic, Train/Test Split)\n\n")
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
