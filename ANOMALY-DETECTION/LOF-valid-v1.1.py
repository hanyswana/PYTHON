#!/usr/bin/env python3
"""
Validate saved PyOD LOF models (trained per-preprocess) using ONE matching validation CSV per model.

Matching rule:
- Model tag = folder name under MODEL_DIR_BASE:
    .../LOF/model/v1.1/<tag>/lof_pyod_model_best.joblib
- Validation CSV filename must be:
    Lablink_330_validation_<tag>.csv
  inside VALID_CSV_DIR

Outputs:
- Per model:
    .../LOF/validation-output/v1.1/<model_tag>/
        - validation_results.csv
        - validation_summary.txt
        - _DONE.ok
        - error_log.txt (if failed)
- Global summary CSV:
    .../LOF/validation-output/v1.1/summary_all_validations.csv

Resume/Skip:
- If _DONE.ok exists (or key outputs exist), it will skip that model.
"""

import os
import sys
import glob
import time
import traceback
import joblib
import json
import numpy as np
import pandas as pd


# ----------------------------
# Base paths
# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"

MODEL_DIR_BASE = f"{DIR}/LOF/model/v1.1"

# Folder containing all validation CSVs
VALID_CSV_DIR = f"{DIR}/dataset-glucose/24/validate/pp"

OUT_DIR_BASE = f"{DIR}/LOF/validation-output/v1.1"

START_COL = "415 nm"
END_COL = "940 nm"
META_CANDIDATES = ["Sample", "glucose"]

VALID_PREFIX = "Lablink_330_validation_"


# ----------------------------
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except BrokenPipeError:
        pass


def select_spectral_columns_by_range(df: pd.DataFrame, start_col=START_COL, end_col=END_COL) -> pd.DataFrame:
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


def model_tag_from_path(model_joblib_path: str) -> str:
    """.../LOF/model/v1.1/<tag>/lof_pyod_model_best.joblib -> <tag>"""
    return os.path.basename(os.path.dirname(model_joblib_path))


def done_marker_path(out_dir: str) -> str:
    return os.path.join(out_dir, "_DONE.ok")


def expected_validation_outputs_exist(out_dir: str) -> bool:
    if os.path.exists(done_marker_path(out_dir)):
        return True
    return (
        os.path.exists(os.path.join(out_dir, "validation_results.csv"))
        and os.path.exists(os.path.join(out_dir, "validation_summary.txt"))
    )


def write_done_marker(out_dir: str):
    with open(done_marker_path(out_dir), "w", encoding="utf-8") as f:
        f.write(f"DONE {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def find_all_models(model_dir_base: str) -> list[str]:
    pattern = os.path.join(model_dir_base, "*", "lof_pyod_model_best.joblib")
    return sorted(glob.glob(pattern))


def build_validation_map(valid_dir: str) -> dict:
    """
    Map: tag -> csv_path for files like Lablink_330_validation_<tag>.csv
    If duplicates exist for same tag, keep the first and warn.
    """
    mp = {}
    csvs = sorted(glob.glob(os.path.join(valid_dir, "*.csv")))
    for p in csvs:
        name = os.path.basename(p)
        if not name.startswith(VALID_PREFIX) or not name.lower().endswith(".csv"):
            continue
        tag = name[len(VALID_PREFIX):-4]  # remove prefix and .csv
        if tag in mp:
            log(f"[WARN] Duplicate validation CSV for tag='{tag}': keeping '{mp[tag]}' and ignoring '{p}'")
            continue
        mp[tag] = p
    return mp


def validate_one(model_path: str, data_path: str) -> dict:
    """
    Validate one LOF model on its matched dataset, save outputs, return summary row.
    """
    t0 = time.time()
    model_tag = model_tag_from_path(model_path)

    out_dir = os.path.join(OUT_DIR_BASE, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    out_results_csv = os.path.join(out_dir, "validation_results.csv")
    out_summary_txt = os.path.join(out_dir, "validation_summary.txt")

    log("=" * 90)
    log(f"[START] model_tag={model_tag}")
    log(f"[INFO ] Model: {model_path}")
    log(f"[INFO ] Data : {data_path}")
    log(f"[INFO ] Out  : {out_dir}")

    # Load model bundle
    bundle = joblib.load(model_path)
    clf = bundle.get("model", None)
    spectral_cols_saved = bundle.get("spectral_columns", None)
    config = bundle.get("config", {})

    if clf is None:
        raise ValueError("Joblib bundle missing key 'model' (expected dict with keys: model, spectral_columns, config).")

    # Load validation data
    df = pd.read_csv(data_path)

    # Metadata (optional)
    meta_cols = [c for c in META_CANDIDATES if c in df.columns]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Extract spectral + clean
    X = select_spectral_columns_by_range(df, START_COL, END_COL)
    X = X.apply(pd.to_numeric, errors="coerce")

    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    dropped = int((~mask_finite).sum())

    X_clean = X.loc[mask_finite].copy()
    meta_clean = meta_df.loc[mask_finite].reset_index(drop=True)

    if dropped > 0:
        log(f"[WARN] Dropped {dropped} rows due to NaN/inf in spectral columns.")

    # Align columns to training order
    if spectral_cols_saved is not None and len(spectral_cols_saved) > 0:
        missing = [c for c in spectral_cols_saved if c not in X_clean.columns]
        if missing:
            raise ValueError(
                "Validation CSV is missing spectral columns used in training:\n"
                + "\n".join(missing[:60])
                + ("\n...(more)" if len(missing) > 60 else "")
            )
        X_aligned = X_clean[spectral_cols_saved].copy()
    else:
        X_aligned = X_clean.copy()

    X_np = X_aligned.to_numpy(dtype=np.float32)

    # Predict + score (PyOD convention)
    # predict: 0=pass, 1=fail
    # decision_function: higher = more anomalous
    y_pred = clf.predict(X_np)
    scores = clf.decision_function(X_np)
    status = np.where(y_pred == 1, "fail", "pass")

    # Save results CSV
    results = pd.DataFrame()
    for c in meta_cols:
        results[c] = meta_clean[c].values
    results["is_anomaly"] = y_pred
    results["status"] = status
    results["anomaly_score"] = scores
    results.to_csv(out_results_csv, index=False)

    # Summary stats
    total = int(len(results))
    fail_count = int((y_pred == 1).sum())
    pass_count = total - fail_count
    fail_rate = (fail_count / total) * 100.0 if total > 0 else 0.0

    score_stats = {}
    if total > 0:
        score_stats = {
            "min": float(np.min(scores)),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(np.max(scores)),
        }

    with open(out_summary_txt, "w", encoding="utf-8") as f:
        f.write("LOF Validation (PyOD) - v1.1 (1 model ↔ 1 dataset)\n")
        f.write("-------------------------------------------------\n")
        f.write(f"model_tag: {model_tag}\n")
        f.write(f"matched_dataset: {os.path.basename(data_path)}\n\n")

        f.write(f"Model: {model_path}\n")
        f.write(f"Data : {data_path}\n")
        f.write(f"Out  : {out_dir}\n\n")

        f.write("Model config (from joblib):\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Features:\n")
        f.write(f"  start='{START_COL}', end='{END_COL}'\n")
        f.write(f"  n_features_used={X_aligned.shape[1]}\n\n")

        f.write("Cleaning:\n")
        f.write(f"  dropped_rows={dropped}\n")
        f.write(f"  used_rows={total}\n\n")

        f.write("Results:\n")
        f.write(f"  pass_count={pass_count}\n")
        f.write(f"  fail_count={fail_count}\n")
        f.write(f"  fail_rate_percent={fail_rate:.3f}\n\n")

        if score_stats:
            f.write("Score stats (higher = more anomalous):\n")
            for k, v in score_stats.items():
                f.write(f"  {k}={v:.6f}\n")

    write_done_marker(out_dir)

    elapsed = time.time() - t0
    log(f"[DONE ] model={model_tag} | used={total} dropped={dropped} fail={fail_count} | {elapsed:.1f}s")

    return {
        "model_tag": model_tag,
        "model_path": model_path,
        "matched_dataset_file": os.path.basename(data_path),
        "dataset_path": data_path,
        "status": "success",
        "error": "",
        "elapsed_sec": round(elapsed, 2),

        "dropped_rows": dropped,
        "used_rows": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "fail_rate_percent": round(fail_rate, 6),

        "score_min": score_stats.get("min", ""),
        "score_p50": score_stats.get("p50", ""),
        "score_p90": score_stats.get("p90", ""),
        "score_p95": score_stats.get("p95", ""),
        "score_p99": score_stats.get("p99", ""),
        "score_max": score_stats.get("max", ""),

        "out_dir": out_dir,
        "results_csv": out_results_csv,
        "summary_txt": out_summary_txt,
        "model_config_json": json.dumps(config, ensure_ascii=False),
    }


def main():
    os.makedirs(OUT_DIR_BASE, exist_ok=True)

    model_paths = find_all_models(MODEL_DIR_BASE)
    if not model_paths:
        raise FileNotFoundError(f"No models found under: {MODEL_DIR_BASE}/*/lof_pyod_model_best.joblib")

    valid_map = build_validation_map(VALID_CSV_DIR)
    if not valid_map:
        raise FileNotFoundError(
            f"No matching validation CSVs found in: {VALID_CSV_DIR}\n"
            f"Expected pattern: {VALID_PREFIX}<tag>.csv"
        )

    log(f"[INFO] Found {len(model_paths)} LOF models in {MODEL_DIR_BASE}")
    log(f"[INFO] Found {len(valid_map)} matching validation CSVs in {VALID_CSV_DIR}")

    summary_rows = []
    summary_csv_path = os.path.join(OUT_DIR_BASE, "summary_all_validations.csv")

    for idx, m in enumerate(model_paths, start=1):
        mtag = model_tag_from_path(m)
        out_dir = os.path.join(OUT_DIR_BASE, mtag)

        log(f"\n[QUEUE] ({idx}/{len(model_paths)}) model={mtag}")

        # Skip if already validated
        if expected_validation_outputs_exist(out_dir):
            log(f"[SKIP] model={mtag} (already completed)")
            summary_rows.append({
                "model_tag": mtag,
                "model_path": m,
                "matched_dataset_file": os.path.basename(valid_map.get(mtag, "")) if valid_map.get(mtag, "") else "",
                "dataset_path": valid_map.get(mtag, ""),
                "status": "skipped",
                "error": "",
                "elapsed_sec": "",
                "out_dir": out_dir,
            })
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            continue

        # Find matched dataset for this model tag
        data_path = valid_map.get(mtag, None)
        if data_path is None:
            err = f"No matched validation CSV for model tag '{mtag}'. Expected: {VALID_PREFIX}{mtag}.csv"
            log(f"[FAIL] {err}")
            summary_rows.append({
                "model_tag": mtag,
                "model_path": m,
                "matched_dataset_file": "",
                "dataset_path": "",
                "status": "failed",
                "error": err,
                "elapsed_sec": "",
                "out_dir": out_dir,
            })
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            continue

        # Validate
        try:
            row = validate_one(m, data_path)
        except Exception as e:
            err = str(e)
            tb = traceback.format_exc()
            os.makedirs(out_dir, exist_ok=True)

            err_path = os.path.join(out_dir, "error_log.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err + "\n\n" + tb)

            log(f"[FAIL] model={mtag} | error={err}")
            row = {
                "model_tag": mtag,
                "model_path": m,
                "matched_dataset_file": os.path.basename(data_path),
                "dataset_path": data_path,
                "status": "failed",
                "error": err,
                "elapsed_sec": "",
                "out_dir": out_dir,
            }

        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
        log(f"[INFO] Updated global summary: {summary_csv_path}")

    log("\nAll validations completed ✅")
    log(f"Global summary saved at: {summary_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
