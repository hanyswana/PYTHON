#!/usr/bin/env python3
"""
Train Mahalanobis Distance (MD) anomaly detector on normal glucose dataset (415..940 nm).

Flow matches IF-TTS:
1) Load + select spectral cols
2) Clean rows (finite)
3) One global 80/20 train-test split
4) Heuristic grid search (TRAIN only) WITHOUT synthetic anomalies
5) Train final MD model on TRAIN only
6) Save model (joblib) + ESP32 header (.h)
7) Save anomaly results for train and test + summary

Outputs are under /MD/ ... (same base path as IF, just IF -> MD)
"""

import os, sys, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, MinCovDet


# ----------------------------
# Paths (same base DIR as IF, but IF -> MD)
# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/raw/Lablink_670_training_normal.csv"

OUT_DIR = f"{DIR}/MD/result/v1.0"
MODEL_DIR = f"{DIR}/MD/model/v1.0"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5
SPLIT_SEEDS = [42, 7, 13, 21, 99]

# Heuristic weights
W_FAILRATE = 1.0
W_SPREAD = 0.10
W_STABILITY = 0.50

# Grid to search (MD-specific)
PARAM_GRID = {
    "contamination": [0.01, 0.02, 0.05],
    # covariance estimator choice:
    "cov_estimator": ["empirical", "ledoitwolf", "oas", "mincovdet"],
    # MinCovDet can be heavy; support_fraction controls robustness (None lets sklearn choose)
    "support_fraction": [None, 0.6, 0.8],
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


def fit_cov_model(X: np.ndarray, cov_estimator: str, support_fraction=None):
    """
    Fit covariance estimator and return (mean, cov, inv_cov).
    """
    if cov_estimator == "empirical":
        est = EmpiricalCovariance(assume_centered=False)
    elif cov_estimator == "ledoitwolf":
        est = LedoitWolf(assume_centered=False)
    elif cov_estimator == "oas":
        est = OAS(assume_centered=False)
    elif cov_estimator == "mincovdet":
        # robust covariance (can be slower)
        est = MinCovDet(assume_centered=False, support_fraction=support_fraction, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown cov_estimator: {cov_estimator}")

    est.fit(X)
    mean = est.location_.astype(np.float32)         # shape (d,)
    cov = est.covariance_.astype(np.float32)        # shape (d,d)
    # numerical safety: invert with pseudo-inverse if needed
    try:
        inv_cov = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov).astype(np.float32)

    return mean, cov, inv_cov, est


def mahalanobis_d2(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis distance squared for each row in X.
    d2 = (x-mean)^T inv_cov (x-mean)
    Returns shape (n,)
    """
    D = (X - mean).astype(np.float32)
    # (n,d) @ (d,d) => (n,d); then elementwise dot with D => (n,)
    tmp = D @ inv_cov
    d2 = np.einsum("ij,ij->i", tmp, D).astype(np.float32)
    # clamp small negatives due to numeric
    d2 = np.maximum(d2, 0.0)
    return d2


def threshold_from_contamination(train_scores: np.ndarray, contamination: float) -> float:
    """
    Threshold on TRAIN scores so that ~contamination fraction are flagged as fail.
    For score = distance^2, higher = more anomalous.
    """
    q = 1.0 - float(contamination)
    thr = float(np.quantile(train_scores, q))
    return thr


def predict_from_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    """
    Return 0=inlier(pass), 1=outlier(fail).
    """
    return (scores > thr).astype(np.int32)


def heuristic_eval_one_split(X_np: np.ndarray, params: dict, seed: int):
    """
    Fit MD on train subset, evaluate on val subset (both normal-only).
    Returns: (val_fail_rate, val_score_std)
    """
    X_train, X_val = train_test_split(X_np, test_size=TEST_SIZE, random_state=seed, shuffle=True)

    mean, cov, inv_cov, _ = fit_cov_model(
        X_train,
        cov_estimator=params["cov_estimator"],
        support_fraction=params["support_fraction"],
    )
    train_scores = mahalanobis_d2(X_train, mean, inv_cov)
    thr = threshold_from_contamination(train_scores, params["contamination"])

    val_scores = mahalanobis_d2(X_val, mean, inv_cov)
    y_val = predict_from_threshold(val_scores, thr)

    val_fail_rate = float(np.mean(y_val == 1))
    val_score_std = float(np.std(val_scores))

    return val_fail_rate, val_score_std


def export_md_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "MD_MODEL_H"):
    """
    Export mean vector, inv_cov matrix, and threshold to a C header for ESP32.
    Inference computes md^2 and compares to threshold.
    """
    bundle = joblib.load(joblib_path)

    mean = bundle["mean"].astype(np.float32)
    inv_cov = bundle["inv_cov"].astype(np.float32)
    thr = float(bundle["threshold"])
    spectral_cols = bundle.get("spectral_columns", [])

    d = int(len(mean))
    if inv_cov.shape != (d, d):
        raise ValueError(f"inv_cov shape mismatch: {inv_cov.shape} expected {(d,d)}")

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("// Auto-generated Mahalanobis Distance model")
    lines.append("// Score: md2 = (x-mean)^T * inv_cov * (x-mean)")
    lines.append("// Decision: fail if md2 > threshold")
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
    lines.append(f"#define MD_N_FEATURES {d}")
    lines.append(f"#define MD_THRESHOLD {thr:.10f}f")
    lines.append("")

    # mean
    lines.append(f"static const float MD_MEAN[{d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in mean))
    lines.append("};\n")

    # inv_cov flattened row-major
    inv_flat = inv_cov.reshape(-1)
    lines.append(f"static const float MD_INV_COV[{d*d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in inv_flat))
    lines.append("};\n")

    lines.append("static inline float md_read_f32(const float* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  union { uint32_t u; float f; } v;")
    lines.append("  v.u = (uint32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("  return v.f;")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline float md_score_md2(const float* x) {")
    lines.append("  // compute d = x - mean")
    lines.append("  float d[MD_N_FEATURES];")
    lines.append("  for (int i=0; i<MD_N_FEATURES; ++i) {")
    lines.append("    d[i] = x[i] - md_read_f32(MD_MEAN, i);")
    lines.append("  }")
    lines.append("  // compute md2 = d^T * inv_cov * d")
    lines.append("  float md2 = 0.0f;")
    lines.append("  for (int i=0; i<MD_N_FEATURES; ++i) {")
    lines.append("    float acc = 0.0f;")
    lines.append("    for (int j=0; j<MD_N_FEATURES; ++j) {")
    lines.append("      acc += md_read_f32(MD_INV_COV, i*MD_N_FEATURES + j) * d[j];")
    lines.append("    }")
    lines.append("    md2 += d[i] * acc;")
    lines.append("  }")
    lines.append("  if (md2 < 0.0f) md2 = 0.0f;")
    lines.append("  return md2;")
    lines.append("}\n")

    lines.append("static inline int md_predict(const float* x) {")
    lines.append("  // return 0=pass(inlier), 1=fail(outlier)")
    lines.append("  float s = md_score_md2(x);")
    lines.append("  return (s > MD_THRESHOLD) ? 1 : 0;")
    lines.append("}\n")

    lines.append(f"#endif // {header_guard}\n")

    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return header_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load + select spectral
    df = pd.read_csv(CSV_PATH)
    X = select_spectral_columns(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    # 2) Clean rows
    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()
    meta_clean = df.loc[mask_finite, ["Sample", "glucose"]].reset_index(drop=True)

    dropped = int((~mask_finite).sum())
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows due to NaN/inf in spectral columns.")

    X_np = X_clean.to_numpy(dtype=np.float32)

    # 3) One global train-test split (80/20)
    X_train, X_test, meta_train, meta_test = train_test_split(
        X_np,
        meta_clean,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"[Split] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4) Heuristic grid search (TRAIN ONLY)
    results = []
    best_obj = float("inf")
    best_params = None

    for params in ParameterGrid(PARAM_GRID):
        # Skip useless support_fraction combos for non-mincovdet
        if params["cov_estimator"] != "mincovdet" and params["support_fraction"] is not None:
            continue

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

        row = {
            **params,
            "mean_val_fail_rate": mean_fail,
            "std_val_fail_rate": std_fail,
            "mean_val_score_std": mean_sd,
            "objective": obj,
        }
        results.append(row)

        print(f"OBJ={obj:.6f}  mean_fail={mean_fail:.4f}±{std_fail:.4f}  score_std={mean_sd:.6f}  params={params}")

        if best_params is None or obj < best_obj:
            best_obj = obj
            best_params = params

    results_df = pd.DataFrame(results).sort_values("objective", ascending=True)
    grid_path = os.path.join(OUT_DIR, "gridsearch_results_heuristic.csv")
    results_df.to_csv(grid_path, index=False)

    print("\n[GridSearch] Best params:", best_params)
    print(f"[GridSearch] Best objective: {best_obj:.6f}")
    print(f"[GridSearch] Saved grid results: {grid_path}")

    # 5) Train FINAL MD model (TRAIN ONLY)
    mean, cov, inv_cov, _ = fit_cov_model(
        X_train,
        cov_estimator=best_params["cov_estimator"],
        support_fraction=best_params["support_fraction"],
    )
    train_scores = mahalanobis_d2(X_train, mean, inv_cov)
    threshold = threshold_from_contamination(train_scores, best_params["contamination"])

    # 6) Save model bundle (joblib)
    best_model_path = os.path.join(MODEL_DIR, "md_model_best.joblib")
    joblib.dump(
        {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
            "threshold": float(threshold),
            "spectral_columns": list(X_clean.columns),
            "config": {
                **best_params,
                "random_state": RANDOM_STATE,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "grid_search_mode": "heuristic_train_only",
                "score": "mahalanobis_distance_squared",
            },
        },
        best_model_path,
    )

    # 7) Export header for ESP32
    best_header_path = os.path.join(MODEL_DIR, "md_model_best.h")
    export_md_joblib_to_header(best_model_path, best_header_path, header_guard="MD_MODEL_BEST_H")

    # 8) TRAIN results (80%)
    y_train = predict_from_threshold(train_scores, threshold)
    train_out = pd.DataFrame({
        "Sample": meta_train["Sample"].values,
        "glucose": meta_train["glucose"].values,
        "is_anomaly": y_train,
        "status": np.where(y_train == 1, "fail", "pass"),
        "anomaly_score": train_scores,
        "split": "train",
    })
    train_out.to_csv(os.path.join(OUT_DIR, "anomaly_results_train.csv"), index=False)

    # 9) TEST results (20%, unseen)
    test_scores = mahalanobis_d2(X_test, mean, inv_cov)
    y_test = predict_from_threshold(test_scores, threshold)
    test_out = pd.DataFrame({
        "Sample": meta_test["Sample"].values,
        "glucose": meta_test["glucose"].values,
        "is_anomaly": y_test,
        "status": np.where(y_test == 1, "fail", "pass"),
        "anomaly_score": test_scores,
        "split": "test",
    })
    test_out.to_csv(os.path.join(OUT_DIR, "anomaly_results_test.csv"), index=False)

    # 10) Summary
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
        "threshold_md2": float(threshold),
    }

    summary_path = os.path.join(OUT_DIR, "anomaly_summary_best.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mahalanobis Distance (Heuristic, Train/Test Split)\n\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Output Dir: {OUT_DIR}\n")
        f.write(f"Model Dir: {MODEL_DIR}\n\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples:  {len(X_test)}\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Threshold (md^2): {threshold:.10f}\n\n")
        f.write(f"Train summary: {summary['train']}\n")
        f.write(f"Test summary:  {summary['test']}\n")

    print("\nDone ✅")
    print(f"Saved BEST model:  {best_model_path}")
    print(f"Saved BEST header: {best_header_path}")
    print("Training results  → anomaly_results_train.csv")
    print("Testing results   → anomaly_results_test.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
