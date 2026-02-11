#!/usr/bin/env python3
"""
Train PyOD Isolation Forest (best joblib) on normal glucose dataset.

- TTS
- apply Grid Search
- saved model in joblib and header files
- saved output result of train and test data
"""

import os, sys, joblib, math
import numpy as np
import pandas as pd

from pyod.models.iforest import IForest
from sklearn.model_selection import train_test_split, ParameterGrid


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/raw/Lablink_670_training_normal.csv"

OUT_DIR = f"{DIR}/IF/result/v1.0"
MODEL_DIR = f"{DIR}/IF/model/v1.0"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5                 # do multiple splits to measure stability
SPLIT_SEEDS = [42, 7, 13, 21, 99]  # length must match N_SPLITS

# Heuristic weights (tune if needed)
W_FAILRATE = 1.0
W_SPREAD = 0.10              # smaller weight because score scale depends on params
W_STABILITY = 0.50           # penalize unstable models across splits

# The grid to search (same as yours)
PARAM_GRID = {
    "contamination": [0.01, 0.02, 0.05],
    "n_estimators": [50, 100, 200],
    "max_samples": ["auto", 128, 256],
    "max_features": [0.5, 1.0],
    "bootstrap": [False],
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


def _c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    euler_gamma = 0.5772156649015329
    return 2.0 * (math.log(n - 1.0) + euler_gamma) - (2.0 * (n - 1.0) / n)


def export_pyod_iforest_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "IFOREST_MODEL_H"):
    bundle = joblib.load(joblib_path)
    pyod_model = bundle["model"]
    spectral_cols = bundle.get("spectral_columns", [])
    cfg = bundle.get("config", {})

    if not hasattr(pyod_model, "detector_"):
        raise ValueError("This PyOD IForest object has no detector_. Did you fit() before saving?")

    sk_if = pyod_model.detector_
    if not hasattr(sk_if, "estimators_"):
        raise ValueError("Underlying sklearn IsolationForest has no estimators_.")

    n_estimators = len(sk_if.estimators_)
    max_samples = int(getattr(sk_if, "max_samples_", cfg.get("max_samples", 256)))
    c = _c_factor(max_samples)

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("// Auto-generated from PyOD IForest (sklearn IsolationForest trees)")
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
    lines.append(f"#define IFOREST_N_FEATURES {len(spectral_cols)}")
    lines.append(f"#define IFOREST_N_ESTIMATORS {n_estimators}")
    lines.append(f"#define IFOREST_MAX_SAMPLES {max_samples}")
    lines.append(f"#define IFOREST_C_FACTOR {c:.10f}f")
    lines.append("")

    tree_sizes = []
    for i, est in enumerate(sk_if.estimators_):
        t = est.tree_
        n_nodes = int(t.node_count)
        tree_sizes.append(n_nodes)

        feat = t.feature.astype(np.int32)
        thr = t.threshold.astype(np.float32)
        left = t.children_left.astype(np.int32)
        right = t.children_right.astype(np.int32)

        lines.append(f"// ---- Tree {i}: nodes={n_nodes} ----")
        lines.append(f"static const int32_t IFOREST_T{i}_FEATURE[{n_nodes}] PROGMEM = " + "{")
        lines.append(",".join(str(int(x)) for x in feat))
        lines.append("};")

        lines.append(f"static const float IFOREST_T{i}_THRESH[{n_nodes}] PROGMEM = " + "{")
        lines.append(",".join(f"{float(x):.7g}f" for x in thr))
        lines.append("};")

        lines.append(f"static const int32_t IFOREST_T{i}_LEFT[{n_nodes}] PROGMEM = " + "{")
        lines.append(",".join(str(int(x)) for x in left))
        lines.append("};")

        lines.append(f"static const int32_t IFOREST_T{i}_RIGHT[{n_nodes}] PROGMEM = " + "{")
        lines.append(",".join(str(int(x)) for x in right))
        lines.append("};\n")

    lines.append("static inline int32_t if_read_i32(const int32_t* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  return (int32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline float if_read_f32(const float* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  union { uint32_t u; float f; } v;")
    lines.append("  v.u = (uint32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("  return v.f;")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline float if_c_factor(int n) {")
    lines.append("  if (n <= 1) return 0.0f;")
    lines.append("  const float euler_gamma = 0.5772156649f;")
    lines.append("  return 2.0f * (logf((float)(n - 1)) + euler_gamma) - (2.0f * (float)(n - 1) / (float)n);")
    lines.append("}\n")

    lines.append("static inline float if_tree_path_length(")
    lines.append("  const float* x,")
    lines.append("  const int32_t* feat, const float* thr, const int32_t* left, const int32_t* right,")
    lines.append("  int32_t n_nodes")
    lines.append(") {")
    lines.append("  int32_t node = 0;")
    lines.append("  float depth = 0.0f;")
    lines.append("  while (node >= 0 && node < n_nodes) {")
    lines.append("    int32_t f = if_read_i32(feat, node);")
    lines.append("    if (f < 0) break;")
    lines.append("    float t = if_read_f32(thr, node);")
    lines.append("    float xv = x[f];")
    lines.append("    int32_t next = (xv <= t) ? if_read_i32(left, node) : if_read_i32(right, node);")
    lines.append("    depth += 1.0f;")
    lines.append("    if (next == -1) break;")
    lines.append("    node = next;")
    lines.append("  }")
    lines.append("  return depth;")
    lines.append("}\n")

    lines.append("static inline float iforest_score(const float* x) {")
    lines.append("  float sum_depth = 0.0f;")
    for i, n_nodes in enumerate(tree_sizes):
        lines.append(
            f"  sum_depth += if_tree_path_length(x, IFOREST_T{i}_FEATURE, IFOREST_T{i}_THRESH, "
            f"IFOREST_T{i}_LEFT, IFOREST_T{i}_RIGHT, {n_nodes});"
        )
    lines.append("  float eh = sum_depth / (float)IFOREST_N_ESTIMATORS;")
    lines.append("  float cn = IFOREST_C_FACTOR;")
    lines.append("  if (cn <= 0.0f) cn = if_c_factor(IFOREST_MAX_SAMPLES);")
    lines.append("  return powf(2.0f, -eh / cn);")
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

    clf = IForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        contamination=params["contamination"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train)

    # On val normals:
    # - predict(X): 0=inlier, 1=outlier (threshold comes from contamination)
    # - decision_function(X): higher => more anomalous
    y_val = clf.predict(X_val)
    val_fail_rate = float(np.mean(y_val == 1))

    val_scores = clf.decision_function(X_val)
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

    # -------------------------------------------------
    # 4) Train FINAL model (TRAIN SET ONLY)
    # -------------------------------------------------
    best_model = IForest(
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        contamination=best_params["contamination"],
        max_features=best_params["max_features"],
        bootstrap=best_params["bootstrap"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    best_model.fit(X_train)

    # -------------------------------------------------
    # 5) Save model + ESP32 header
    # -------------------------------------------------
    best_model_path = os.path.join(MODEL_DIR, "iforest_pyod_model_best.joblib")
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

    best_header_path = os.path.join(MODEL_DIR, "iforest_pyod_model_best.h")
    export_pyod_iforest_joblib_to_header(
        best_model_path,
        best_header_path,
        header_guard="IFOREST_PYOD_MODEL_BEST_H",
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

    train_results.to_csv(
        os.path.join(OUT_DIR, "anomaly_results_train.csv"),
        index=False,
    )

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

    test_results.to_csv(
        os.path.join(OUT_DIR, "anomaly_results_test.csv"),
        index=False,
    )

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
        f.write("Isolation Forest (Heuristic, Train/Test Split)\n\n")
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
