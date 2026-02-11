#!/usr/bin/env python3
"""
Train Mahalanobis Distance (MD) anomaly detector on normal glucose dataset (415..940 nm).

UPDATED to match IF v1.1 flow:
- Safe logger (won't crash if stdout pipe breaks)
- Resume/Skip: if dataset already completed (DONE marker OR key outputs exist), it will be skipped
- DONE marker is written after a successful run
- Global summary CSV updated after each dataset
"""

import os, sys, joblib, glob, time, json, traceback
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, MinCovDet


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"

CSV_PATH = f"{DIR}/dataset-glucose/24/train/pp"

OUT_DIR_BASE = f"{DIR}/MD/result/v1.1"   # keep same as v1.1 unless you want new folder
MODEL_DIR_BASE = f"{DIR}/MD/model/v1.1"  # keep same as v1.1 unless you want new folder

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
    "cov_estimator": ["empirical", "ledoitwolf", "oas", "mincovdet"],
    "support_fraction": [None, 0.6, 0.8],

}

# Naming convention anchor (same rule as IF)
BASE_NAME = "Lablink_670_training_normal"


# ----------------------------
def log(msg: str):
    """Timestamped logger that won't crash if stdout pipe breaks (e.g., PyCharm console closes)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except BrokenPipeError:
        pass


def done_marker_path(out_dir: str) -> str:
    return os.path.join(out_dir, "_DONE.ok")


def write_done_marker(out_dir: str):
    with open(done_marker_path(out_dir), "w", encoding="utf-8") as f:
        f.write(f"DONE {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def expected_outputs_exist(out_dir: str, model_dir: str) -> bool:
    """
    Strong check: either DONE marker exists OR key output files exist.
    Matches IF v1.1 logic, but MD filenames.
    """
    if os.path.exists(done_marker_path(out_dir)):
        return True

    common = [
        os.path.join(out_dir, "gridsearch_results_heuristic.csv"),
        os.path.join(out_dir, "anomaly_results_train.csv"),
        os.path.join(out_dir, "anomaly_results_test.csv"),
        os.path.join(out_dir, "anomaly_summary_best.txt"),
    ]
    model_files = [
        os.path.join(model_dir, "md_model_best.joblib"),
        os.path.join(model_dir, "md_model_best.h"),
    ]
    return all(os.path.exists(p) for p in (common + model_files))


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
    return df[cols[i0: i1 + 1]].copy()


def dataset_suffix_from_filename(csv_file: str) -> str:
    """
    Extract suffix after BASE_NAME from filename.
    Examples:
      Lablink_670_training_normal.csv -> "raw"
      Lablink_670_training_normal_SNV_Baseline.csv -> "SNV_Baseline"
    """
    name = os.path.basename(csv_file)
    stem = name[:-4] if name.lower().endswith(".csv") else name

    if stem == BASE_NAME:
        return "raw"
    if stem.startswith(BASE_NAME + "_"):
        return stem[len(BASE_NAME) + 1:].strip() or "raw"
    return stem  # fallback


def fit_cov_model(X: np.ndarray, cov_estimator: str, support_fraction=None):
    if cov_estimator == "empirical":
        est = EmpiricalCovariance(assume_centered=False)
    elif cov_estimator == "ledoitwolf":
        est = LedoitWolf(assume_centered=False)
    elif cov_estimator == "oas":
        est = OAS(assume_centered=False)
    elif cov_estimator == "mincovdet":
        est = MinCovDet(
            assume_centered=False,
            support_fraction=support_fraction,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown cov_estimator: {cov_estimator}")

    est.fit(X)
    mean = est.location_.astype(np.float32)
    cov = est.covariance_.astype(np.float32)
    try:
        inv_cov = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov).astype(np.float32)

    return mean, cov, inv_cov, est


def mahalanobis_d2(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    D = (X - mean).astype(np.float32)
    tmp = D @ inv_cov
    d2 = np.einsum("ij,ij->i", tmp, D).astype(np.float32)
    return np.maximum(d2, 0.0)


def threshold_from_contamination(train_scores: np.ndarray, contamination: float) -> float:
    q = 1.0 - float(contamination)
    return float(np.quantile(train_scores, q))


def predict_from_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores > thr).astype(np.int32)  # 0=pass, 1=fail


def heuristic_eval_one_split(X_np: np.ndarray, params: dict, seed: int):
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
    bundle = joblib.load(joblib_path)

    mean = bundle["mean"].astype(np.float32)
    inv_cov = bundle["inv_cov"].astype(np.float32)
    thr = float(bundle["threshold"])

    d = int(len(mean))
    if inv_cov.shape != (d, d):
        raise ValueError(f"inv_cov shape mismatch: {inv_cov.shape} expected {(d, d)}")

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

    lines.append(f"static const float MD_MEAN[{d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in mean))
    lines.append("};\n")

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
    lines.append("  float d[MD_N_FEATURES];")
    lines.append("  for (int i=0; i<MD_N_FEATURES; ++i) {")
    lines.append("    d[i] = x[i] - md_read_f32(MD_MEAN, i);")
    lines.append("  }")
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
    lines.append("  float s = md_score_md2(x);")
    lines.append("  return (s > MD_THRESHOLD) ? 1 : 0;")
    lines.append("}\n")

    lines.append(f"#endif // {header_guard}\n")

    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return header_path


def run_one_dataset(csv_file: str) -> dict:
    t0 = time.time()
    tag = dataset_suffix_from_filename(csv_file)

    out_dir = os.path.join(OUT_DIR_BASE, tag)
    model_dir = os.path.join(MODEL_DIR_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log("=" * 90)
    log(f"[START] Dataset: {os.path.basename(csv_file)}")
    log(f"[INFO ] Tag: {tag}")
    log(f"[INFO ] OUT : {out_dir}")
    log(f"[INFO ] MODEL: {model_dir}")

    # Stage 1: load
    log("[STAGE 1/6] Loading CSV...")
    df = pd.read_csv(csv_file)
    n_rows_raw = int(len(df))

    # Stage 2: build X
    log("[STAGE 2/6] Selecting spectral columns + cleaning...")
    X = select_spectral_columns(df, "415 nm", "940 nm")
    X = X.apply(pd.to_numeric, errors="coerce")

    mask_finite = np.isfinite(X.to_numpy()).all(axis=1)
    X_clean = X.loc[mask_finite].copy()
    n_rows_clean = int(len(X_clean))
    n_features = int(X_clean.shape[1])

    required_meta = ["Sample", "glucose"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing required columns: {missing_meta}")

    meta_clean = df.loc[mask_finite, required_meta].reset_index(drop=True)
    X_np = X_clean.to_numpy(dtype=np.float32)

    log(f"[INFO ] Rows raw={n_rows_raw}, clean={n_rows_clean}, dropped={n_rows_raw - n_rows_clean}, features={n_features}")

    # Stage 3: split
    log("[STAGE 3/6] Train/Test split...")
    X_train, X_test, meta_train, meta_test = train_test_split(
        X_np, meta_clean, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    n_train = int(len(X_train))
    n_test = int(len(X_test))
    log(f"[INFO ] Split -> train={n_train}, test={n_test}")

    # Stage 4: grid search
    log("[STAGE 4/6] Heuristic grid search...")
    results = []
    best_obj = float("inf")
    best_params = None

    valid_grid = []
    for p in ParameterGrid(PARAM_GRID):
        if p["cov_estimator"] != "mincovdet" and p["support_fraction"] is not None:
            continue
        valid_grid.append(p)

    total_candidates = len(valid_grid)
    cand_idx = 0

    for params in valid_grid:
        cand_idx += 1
        log(f"[GRID ] Candidate {cand_idx}/{total_candidates}: {params}")

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
            log(f"[BEST ] Updated best objective={best_obj:.6f} with params={best_params}")

    results_df = pd.DataFrame(results).sort_values("objective", ascending=True)
    grid_path = os.path.join(out_dir, "gridsearch_results_heuristic.csv")
    results_df.to_csv(grid_path, index=False)
    log(f"[INFO ] Grid results saved: {grid_path}")
    log(f"[INFO ] Best params: {best_params}")

    # Stage 5: train final
    log("[STAGE 5/6] Training final best MD model...")
    mean, cov, inv_cov, _ = fit_cov_model(
        X_train,
        cov_estimator=best_params["cov_estimator"],
        support_fraction=best_params["support_fraction"],
    )
    train_scores = mahalanobis_d2(X_train, mean, inv_cov)
    threshold = threshold_from_contamination(train_scores, best_params["contamination"])

    # Stage 6: save outputs
    log("[STAGE 6/6] Saving model + header + results...")

    best_model_path = os.path.join(model_dir, "md_model_best.joblib")
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
                "train_size": n_train,
                "test_size": n_test,
                "grid_search_mode": "heuristic_train_only",
                "score": "mahalanobis_distance_squared",
                "dataset_file": os.path.basename(csv_file),
                "dataset_tag": tag,
            },
        },
        best_model_path,
    )

    best_header_path = os.path.join(model_dir, "md_model_best.h")
    export_md_joblib_to_header(
        best_model_path,
        best_header_path,
        header_guard=f"MD_MODEL_BEST_H_{tag.upper()}".replace("-", "_"),
    )

    # Train results
    y_train = predict_from_threshold(train_scores, threshold)
    train_fail = int((y_train == 1).sum())
    train_pass = int((y_train == 0).sum())
    train_fail_rate = float(np.mean(y_train == 1) * 100.0)

    train_out = pd.DataFrame({
        "Sample": meta_train["Sample"].values,
        "glucose": meta_train["glucose"].values,
        "is_anomaly": y_train,
        "status": np.where(y_train == 1, "fail", "pass"),
        "anomaly_score": train_scores,
        "split": "train",
    })
    train_csv_path = os.path.join(out_dir, "anomaly_results_train.csv")
    train_out.to_csv(train_csv_path, index=False)

    # Test results
    test_scores = mahalanobis_d2(X_test, mean, inv_cov)
    y_test = predict_from_threshold(test_scores, threshold)
    test_fail = int((y_test == 1).sum())
    test_pass = int((y_test == 0).sum())
    test_fail_rate = float(np.mean(y_test == 1) * 100.0)

    test_out = pd.DataFrame({
        "Sample": meta_test["Sample"].values,
        "glucose": meta_test["glucose"].values,
        "is_anomaly": y_test,
        "status": np.where(y_test == 1, "fail", "pass"),
        "anomaly_score": test_scores,
        "split": "test",
    })
    test_csv_path = os.path.join(out_dir, "anomaly_results_test.csv")
    test_out.to_csv(test_csv_path, index=False)

    # Text summary
    summary_path = os.path.join(out_dir, "anomaly_summary_best.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mahalanobis Distance (Heuristic, Train/Test Split)\n\n")
        f.write(f"Dataset file: {os.path.basename(csv_file)}\n")
        f.write(f"Dataset tag:  {tag}\n\n")
        f.write(f"Rows raw:   {n_rows_raw}\n")
        f.write(f"Rows clean: {n_rows_clean}\n")
        f.write(f"Features:   {n_features}\n\n")
        f.write(f"Train samples: {n_train}\n")
        f.write(f"Test samples:  {n_test}\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Threshold (md^2): {threshold:.10f}\n\n")
        f.write(f"Train fail/pass: {train_fail}/{train_pass} ({train_fail_rate:.2f}%)\n")
        f.write(f"Test  fail/pass: {test_fail}/{test_pass} ({test_fail_rate:.2f}%)\n")

    elapsed = time.time() - t0
    log(f"[DONE ] {tag} | elapsed={elapsed:.1f}s | thr={threshold:.6f} | train_fail={train_fail} | test_fail={test_fail}")

    # ✅ IMPORTANT: mark done so it can resume/skip next run
    write_done_marker(out_dir)

    return {
        "dataset_file": os.path.basename(csv_file),
        "dataset_path": csv_file,
        "tag": tag,
        "status": "success",
        "error": "",
        "elapsed_sec": round(elapsed, 2),

        "rows_raw": n_rows_raw,
        "rows_clean": n_rows_clean,
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,

        "best_params_json": json.dumps(best_params, ensure_ascii=False),
        "threshold_md2": float(threshold),

        "train_fail": train_fail,
        "train_pass": train_pass,
        "train_fail_rate_percent": round(train_fail_rate, 4),

        "test_fail": test_fail,
        "test_pass": test_pass,
        "test_fail_rate_percent": round(test_fail_rate, 4),

        "out_dir": out_dir,
        "model_dir": model_dir,
        "gridsearch_csv": grid_path,
        "train_results_csv": train_csv_path,
        "test_results_csv": test_csv_path,
        "model_joblib": best_model_path,
        "model_header": best_header_path,
        "text_summary": summary_path,
    }


def main():
    os.makedirs(OUT_DIR_BASE, exist_ok=True)
    os.makedirs(MODEL_DIR_BASE, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(CSV_PATH, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {CSV_PATH}")

    log(f"[INFO] Found {len(csv_files)} CSV files in {CSV_PATH}")

    summary_rows = []
    summary_csv_path = os.path.join(OUT_DIR_BASE, "summary_all_datasets.csv")

    for i, csv_file in enumerate(csv_files, start=1):
        tag = dataset_suffix_from_filename(csv_file)
        out_dir = os.path.join(OUT_DIR_BASE, tag)
        model_dir = os.path.join(MODEL_DIR_BASE, tag)

        log(f"\n[QUEUE] ({i}/{len(csv_files)}) -> {os.path.basename(csv_file)}")

        # ✅ SKIP BLOCK (matches IF flow)
        if expected_outputs_exist(out_dir, model_dir):
            log(f"[SKIP] {os.path.basename(csv_file)} | tag={tag} (already completed)")

            summary_rows.append({
                "dataset_file": os.path.basename(csv_file),
                "dataset_path": csv_file,
                "tag": tag,
                "status": "skipped",
                "error": "",
                "elapsed_sec": "",
                "out_dir": out_dir,
                "model_dir": model_dir,
            })

            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            continue

        try:
            row = run_one_dataset(csv_file)
        except Exception as e:
            err = str(e)
            tb = traceback.format_exc()
            os.makedirs(out_dir, exist_ok=True)

            err_path = os.path.join(out_dir, "error_log.txt")
            with open(err_path, "w") as f:
                f.write(err + "\n\n" + tb)

            log(f"[FAIL] {os.path.basename(csv_file)} | tag={tag} | error={err}")

            row = {
                "dataset_file": os.path.basename(csv_file),
                "dataset_path": csv_file,
                "tag": tag,
                "status": "failed",
                "error": err,
                "elapsed_sec": "",

                "rows_raw": "",
                "rows_clean": "",
                "n_features": "",
                "n_train": "",
                "n_test": "",

                "best_params_json": "",
                "threshold_md2": "",

                "train_fail": "",
                "train_pass": "",
                "train_fail_rate_percent": "",
                "test_fail": "",
                "test_pass": "",
                "test_fail_rate_percent": "",

                "out_dir": out_dir,
                "model_dir": model_dir,
                "gridsearch_csv": "",
                "train_results_csv": "",
                "test_results_csv": "",
                "model_joblib": "",
                "model_header": "",
                "text_summary": "",
            }

        summary_rows.append(row)

        # Save running global summary after each dataset
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
        log(f"[INFO] Updated global summary: {summary_csv_path}")

    log("\nAll datasets completed ✅")
    log(f"Global summary saved at: {summary_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
