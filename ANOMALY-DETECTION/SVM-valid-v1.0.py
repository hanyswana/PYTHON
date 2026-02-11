#!/usr/bin/env python3
"""
Validate saved PyOD One-Class SVM (OCSVM) (best joblib) on NEW dataset.

- Model:   /home/apc-3/PycharmProjects/PythonProjectAD/SVM/model/v1.0/ocsvm_pyod_model_best.joblib
- Dataset: /home/apc-3/PycharmProjects/PythonProjectAD/dataset-glucose/raw/Lablink_330_validation.csv

Outputs:
- /home/apc-3/PycharmProjects/PythonProjectAD/SVM/validation-output/v1.0/validation_results.csv
- /home/apc-3/PycharmProjects/PythonProjectAD/SVM/validation-output/v1.0/validation_summary.txt
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Paths (same base DIR as training)
# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"

MODEL_PATH = f"{DIR}/SVM/model/v1.0/ocsvm_pyod_model_best.joblib"
DATA_PATH  = f"{DIR}/dataset-glucose/raw/Lablink_330_validation.csv"

OUT_DIR = f"{DIR}/SVM/validation-output/v1.0"
OUT_RESULTS_CSV = os.path.join(OUT_DIR, "validation_results.csv")
OUT_SUMMARY_TXT = os.path.join(OUT_DIR, "validation_summary.txt")


def select_spectral_columns_by_range(df: pd.DataFrame, start_col="415 nm", end_col="940 nm") -> pd.DataFrame:
    cols = list(df.columns)
    if start_col not in cols or end_col not in cols:
        raise ValueError(
            f"Missing spectral range columns '{start_col}' or '{end_col}'.\n"
            f"First columns: {cols[:10]}\nLast columns:  {cols[-10:]}"
        )
    i0 = cols.index(start_col)
    i1 = cols.index(end_col)
    if i1 < i0:
        raise ValueError(f"Column '{end_col}' appears before '{start_col}'. Check CSV column order.")
    return df[cols[i0:i1 + 1]].copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load model bundle
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model joblib not found:\n  {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    clf = bundle.get("model", None)
    spectral_cols_saved = bundle.get("spectral_columns", None)
    config = bundle.get("config", {})

    if clf is None:
        raise ValueError("Joblib bundle missing key 'model'. Expected a dict with keys: model, spectral_columns, config.")

    # 2) Load validation dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Validation CSV not found:\n  {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 3) Get metadata columns (Sample, glucose) if present
    meta_cols = []
    for c in ["Sample", "glucose"]:
        if c in df.columns:
            meta_cols.append(c)

    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # 4) Extract spectral range 415..940
    X = select_spectral_columns_by_range(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    # 5) Drop rows with NaN/inf in spectral
    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()
    meta_clean = meta_df.loc[mask_finite].reset_index(drop=True)

    dropped = int((~mask_finite).sum())
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows due to NaN/inf in spectral columns.")

    # 6) Align feature columns to training order
    if spectral_cols_saved is not None and len(spectral_cols_saved) > 0:
        missing = [c for c in spectral_cols_saved if c not in X_clean.columns]
        if missing:
            raise ValueError(
                "Validation CSV is missing spectral columns used in training:\n"
                + "\n".join(missing[:50])
                + ("\n...(more)" if len(missing) > 50 else "")
            )
        X_aligned = X_clean[spectral_cols_saved].copy()
    else:
        X_aligned = X_clean.copy()

    X_np = X_aligned.to_numpy(dtype=np.float32)

    # 7) Predict + score
    # PyOD convention:
    # - predict: 0=inlier(pass), 1=outlier(fail)
    # - decision_function: higher = more anomalous
    y_pred = clf.predict(X_np)
    scores = clf.decision_function(X_np)

    status = np.where(y_pred == 1, "fail", "pass")

    # 8) Save results CSV (include Sample + glucose when available)
    results = pd.DataFrame()

    if "Sample" in meta_clean.columns:
        results["Sample"] = meta_clean["Sample"].values
    if "glucose" in meta_clean.columns:
        results["glucose"] = meta_clean["glucose"].values

    results["is_anomaly"] = y_pred
    results["status"] = status
    results["anomaly_score"] = scores

    results.to_csv(OUT_RESULTS_CSV, index=False)

    # 9) Save summary
    total = int(len(results))
    fail_count = int((y_pred == 1).sum())
    pass_count = total - fail_count
    fail_rate = (fail_count / total) * 100.0 if total > 0 else 0.0

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("One-Class SVM Validation (PyOD OCSVM)\n")
        f.write("-----------------------------------\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data:  {DATA_PATH}\n")
        f.write(f"Out:   {OUT_DIR}\n\n")

        f.write("Model config (from joblib):\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Features:\n")
        f.write("  start='415 nm', end='940 nm'\n")
        f.write(f"  n_features_used={X_aligned.shape[1]}\n\n")

        f.write("Results:\n")
        f.write(f"  total_used_rows={total}\n")
        f.write(f"  pass_count={pass_count}\n")
        f.write(f"  fail_count={fail_count}\n")
        f.write(f"  fail_rate_percent={fail_rate:.3f}\n\n")

        if total > 0:
            f.write("Score stats (higher = more anomalous):\n")
            f.write(f"  min={float(np.min(scores)):.6f}\n")
            f.write(f"  p50={float(np.percentile(scores, 50)):.6f}\n")
            f.write(f"  p90={float(np.percentile(scores, 90)):.6f}\n")
            f.write(f"  p95={float(np.percentile(scores, 95)):.6f}\n")
            f.write(f"  p99={float(np.percentile(scores, 99)):.6f}\n")
            f.write(f"  max={float(np.max(scores)):.6f}\n")

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
