#!/usr/bin/env python3
"""
Validate saved PCA anomaly detector (T² + Q) on NEW dataset.

- Model:   /home/apc-3/PycharmProjects/PythonProjectAD/PCA/model/v1.0/pca_model_best.joblib
- Dataset: /home/apc-3/PycharmProjects/PythonProjectAD/dataset-glucose/raw/Lablink_330_validation.csv

Outputs:
- /home/apc-3/PycharmProjects/PythonProjectAD/PCA/validation-output/v1.0/validation_results.csv
- /home/apc-3/PycharmProjects/PythonProjectAD/PCA/validation-output/v1.0/validation_summary.txt
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd


DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"

MODEL_PATH = f"{DIR}/PCA/model/v1.0/pca_model_best.joblib"
DATA_PATH  = f"{DIR}/dataset-glucose/raw/Lablink_330_validation.csv"

OUT_DIR = f"{DIR}/PCA/validation-output/v1.0"
OUT_RESULTS_CSV = os.path.join(OUT_DIR, "validation_results.csv")
OUT_SUMMARY_TXT = os.path.join(OUT_DIR, "validation_summary.txt")


def select_spectral_columns_by_range(df: pd.DataFrame, start_col="415 nm", end_col="940 nm") -> pd.DataFrame:
    cols = list(df.columns)
    if start_col not in cols or end_col not in cols:
        raise ValueError(f"Missing spectral range columns '{start_col}' or '{end_col}'.")
    i0 = cols.index(start_col)
    i1 = cols.index(end_col)
    if i1 < i0:
        raise ValueError(f"Column '{end_col}' appears before '{start_col}'.")
    return df[cols[i0:i1 + 1]].copy()


def pca_scores_T2_Q(X: np.ndarray, mean: np.ndarray, components: np.ndarray, eigvals: np.ndarray):
    Xc = (X - mean).astype(np.float32)
    z = Xc @ components.T
    T2 = np.sum((z * z) / eigvals, axis=1).astype(np.float32)

    Xhat = mean + (z @ components)
    r = (X - Xhat).astype(np.float32)
    Q = np.sum(r * r, axis=1).astype(np.float32)

    return np.maximum(T2, 0.0), np.maximum(Q, 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model joblib not found:\n  {MODEL_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Validation CSV not found:\n  {DATA_PATH}")

    bundle = joblib.load(MODEL_PATH)

    mean = np.asarray(bundle["mean"], dtype=np.float32)
    comps = np.asarray(bundle["components"], dtype=np.float32)
    eigvals = np.asarray(bundle["eigvals"], dtype=np.float32)
    T2_thr = float(bundle["T2_threshold"])
    Q_thr = float(bundle["Q_threshold"])
    spectral_cols_saved = bundle.get("spectral_columns", [])
    config = bundle.get("config", {})

    df = pd.read_csv(DATA_PATH)

    # metadata
    meta_cols = [c for c in ["Sample", "glucose"] if c in df.columns]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # features
    X = select_spectral_columns_by_range(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    # clean rows
    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()
    meta_clean = meta_df.loc[mask_finite].reset_index(drop=True)

    dropped = int((~mask_finite).sum())
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows due to NaN/inf in spectral columns.")

    # align columns to training order
    if spectral_cols_saved:
        missing = [c for c in spectral_cols_saved if c not in X_clean.columns]
        if missing:
            raise ValueError(f"Validation CSV missing spectral columns used in training: {missing[:10]} ...")
        X_aligned = X_clean[spectral_cols_saved].copy()
    else:
        X_aligned = X_clean.copy()

    X_np = X_aligned.to_numpy(dtype=np.float32)

    # score + predict
    T2, Q = pca_scores_T2_Q(X_np, mean, comps, eigvals)
    y_pred = ((T2 > T2_thr) | (Q > Q_thr)).astype(np.int32)  # 0=pass, 1=fail
    status = np.where(y_pred == 1, "fail", "pass")
    anomaly_score = 0.5 * (T2 / max(T2_thr, 1e-12) + Q / max(Q_thr, 1e-12))

    # save results
    out = pd.DataFrame()
    if "Sample" in meta_clean.columns:
        out["Sample"] = meta_clean["Sample"].values
    if "glucose" in meta_clean.columns:
        out["glucose"] = meta_clean["glucose"].values

    out["is_anomaly"] = y_pred
    out["status"] = status
    out["anomaly_score"] = anomaly_score
    out["T2"] = T2
    out["Q"] = Q

    out.to_csv(OUT_RESULTS_CSV, index=False)

    # summary
    total = int(len(out))
    fail_count = int((y_pred == 1).sum())
    pass_count = total - fail_count
    fail_rate = (fail_count / total) * 100.0 if total else 0.0

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("PCA Validation (T² + Q)\n")
        f.write("-----------------------\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data:  {DATA_PATH}\n")
        f.write(f"Out:   {OUT_DIR}\n\n")

        f.write("Model config (from joblib):\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"T2 threshold: {T2_thr:.10f}\n")
        f.write(f"Q  threshold: {Q_thr:.10f}\n\n")

        f.write("Results:\n")
        f.write(f"  total_used_rows={total}\n")
        f.write(f"  pass_count={pass_count}\n")
        f.write(f"  fail_count={fail_count}\n")
        f.write(f"  fail_rate_percent={fail_rate:.3f}\n\n")

        if total:
            f.write("Score stats (normalized combined anomaly_score):\n")
            f.write(f"  min={float(np.min(anomaly_score)):.6f}\n")
            f.write(f"  p50={float(np.percentile(anomaly_score, 50)):.6f}\n")
            f.write(f"  p90={float(np.percentile(anomaly_score, 90)):.6f}\n")
            f.write(f"  p95={float(np.percentile(anomaly_score, 95)):.6f}\n")
            f.write(f"  p99={float(np.percentile(anomaly_score, 99)):.6f}\n")
            f.write(f"  max={float(np.max(anomaly_score)):.6f}\n")

    print("Done ✅")
    print(f"Saved results: {OUT_RESULTS_CSV}")
    print(f"Saved summary: {OUT_SUMMARY_TXT}")
    print(f"Pass={pass_count}, Fail={fail_count}, Total={total} (Dropped={dropped})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
