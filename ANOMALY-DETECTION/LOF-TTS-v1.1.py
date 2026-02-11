#!/usr/bin/env python3
"""
Train PyOD LOF on normal glucose dataset (415..940 nm).

UPDATED to match IF-TTS-v1.1 flow:
- Use ALL CSV datasets in folder CSV_PATH
- Create output/model folders per dataset tag (suffix after Lablink_670_training_normal)
- Debug stage logs per dataset
- Skip/resume via _DONE.ok + expected outputs check
- Save 1 global summary CSV for all datasets

Based on:
- LOF-TTS-v1.0.py (LOF algorithm + header export) :contentReference[oaicite:4]{index=4}
- IF-TTS-v1.1.py (multi-dataset flow + skip/resume + global summary) :contentReference[oaicite:5]{index=5}
"""

import os, sys, glob, time, json, traceback, joblib
import numpy as np
import pandas as pd

from pyod.models.lof import LOF
from sklearn.model_selection import train_test_split, ParameterGrid


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/24/train/pp"   # folder containing many CSVs

OUT_DIR_BASE = f"{DIR}/LOF/result/v1.1"
MODEL_DIR_BASE = f"{DIR}/LOF/model/v1.1"

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

BASE_NAME = "Lablink_670_training_normal"


# ----------------------------
def log(msg: str):
    """Timestamped logger that won't crash if stdout pipe breaks (PyCharm crash/disconnect)."""
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
    LOF filenames (model/header) differ from IF.
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
        os.path.join(model_dir, "lof_pyod_model_best.joblib"),
        os.path.join(model_dir, "lof_pyod_model_best.h"),
    ]

    return all(os.path.exists(p) for p in (common + model_files))


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
    return df[cols[i0:i1 + 1]].copy()


def export_pyod_lof_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "LOF_MODEL_H"):
    """
    Same as LOF v1.0: export reference set for approximate inference on ESP32.
    WARNING: header can be LARGE because LOF needs training reference points.
    """
    bundle = joblib.load(joblib_path)
    pyod_model = bundle["model"]

    if not hasattr(pyod_model, "detector_"):
        raise ValueError("This PyOD LOF object has no detector_. Did you fit() before saving?")

    det = pyod_model.detector_
    X_ref = getattr(det, "_fit_X", None)
    if X_ref is None:
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
    lines.append("  float best[LOF_K];")
    lines.append("  for (int i=0; i<LOF_K; ++i) best[i] = 1e30f;")
    lines.append("  for (int i=0; i<LOF_N_REF; ++i) {")
    lines.append("    float d2 = lof_dist2(x, i);")
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
        novelty=True,
    )
    clf.fit(X_train)

    y_val = clf.predict(X_val)  # 0=pass, 1=fail
    val_fail_rate = float(np.mean(y_val == 1))

    val_scores = clf.decision_function(X_val)  # higher => more anomalous (PyOD)
    val_score_std = float(np.std(val_scores))

    return val_fail_rate, val_score_std


def run_one_dataset(csv_file: str) -> dict:
    """
    Train LOF for one dataset and return a dict row for the global summary CSV.
    """
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

    total_candidates = len(list(ParameterGrid(PARAM_GRID)))
    cand_idx = 0

    for params in ParameterGrid(PARAM_GRID):
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
    log("[STAGE 5/6] Training final best model...")
    best_model = LOF(
        n_neighbors=best_params["n_neighbors"],
        contamination=best_params["contamination"],
        leaf_size=best_params["leaf_size"],
        metric=best_params["metric"],
        p=best_params["p"],
        novelty=True,
    )
    best_model.fit(X_train)

    # Stage 6: save outputs
    log("[STAGE 6/6] Saving model + header + results...")

    best_model_path = os.path.join(model_dir, "lof_pyod_model_best.joblib")
    joblib.dump(
        {
            "model": best_model,
            "spectral_columns": list(X_clean.columns),
            "config": {
                **best_params,
                "random_state": RANDOM_STATE,
                "train_size": n_train,
                "test_size": n_test,
                "grid_search_mode": "heuristic_train_only",
                "dataset_file": os.path.basename(csv_file),
                "dataset_tag": tag,
            },
        },
        best_model_path,
    )

    best_header_path = os.path.join(model_dir, "lof_pyod_model_best.h")
    export_pyod_lof_joblib_to_header(
        best_model_path,
        best_header_path,
        header_guard=f"LOF_PYOD_MODEL_BEST_H_{tag.upper()}".replace("-", "_"),
    )

    # Train results
    y_train = best_model.predict(X_train)
    scores_train = best_model.decision_function(X_train)
    train_fail = int((y_train == 1).sum())
    train_pass = int((y_train == 0).sum())
    train_fail_rate = float(np.mean(y_train == 1) * 100.0)

    train_results = pd.DataFrame({
        "Sample": meta_train["Sample"].values,
        "glucose": meta_train["glucose"].values,
        "is_anomaly": y_train,
        "status": np.where(y_train == 1, "fail", "pass"),
        "anomaly_score": scores_train,
        "split": "train",
    })
    train_csv_path = os.path.join(out_dir, "anomaly_results_train.csv")
    train_results.to_csv(train_csv_path, index=False)

    # Test results
    y_test = best_model.predict(X_test)
    scores_test = best_model.decision_function(X_test)
    test_fail = int((y_test == 1).sum())
    test_pass = int((y_test == 0).sum())
    test_fail_rate = float(np.mean(y_test == 1) * 100.0)

    test_results = pd.DataFrame({
        "Sample": meta_test["Sample"].values,
        "glucose": meta_test["glucose"].values,
        "is_anomaly": y_test,
        "status": np.where(y_test == 1, "fail", "pass"),
        "anomaly_score": scores_test,
        "split": "test",
    })
    test_csv_path = os.path.join(out_dir, "anomaly_results_test.csv")
    test_results.to_csv(test_csv_path, index=False)

    # Text summary
    summary_path = os.path.join(out_dir, "anomaly_summary_best.txt")
    with open(summary_path, "w") as f:
        f.write("LOF (Heuristic, Train/Test Split)\n\n")
        f.write(f"Dataset file: {os.path.basename(csv_file)}\n")
        f.write(f"Dataset tag:  {tag}\n\n")
        f.write(f"Rows raw:   {n_rows_raw}\n")
        f.write(f"Rows clean: {n_rows_clean}\n")
        f.write(f"Features:   {n_features}\n\n")
        f.write(f"Train samples: {n_train}\n")
        f.write(f"Test samples:  {n_test}\n\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write(f"Train fail/pass: {train_fail}/{train_pass} ({train_fail_rate:.2f}%)\n")
        f.write(f"Test  fail/pass: {test_fail}/{test_pass} ({test_fail_rate:.2f}%)\n")

    elapsed = time.time() - t0
    log(f"[DONE ] {tag} | elapsed={elapsed:.1f}s | train_fail={train_fail} | test_fail={test_fail}")

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

        # Skip already-completed datasets
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
            with open(err_path, "w", encoding="utf-8") as f:
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
