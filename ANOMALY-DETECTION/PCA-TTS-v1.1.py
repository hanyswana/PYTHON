#!/usr/bin/env python3
"""
PCA Anomaly Detection Training (T² + Q/SPE) — v1.1

UPDATED to match IF-TTS-v1.1 flow:
- Use ALL CSV datasets in folder CSV_PATH
- Create output/model folders per dataset tag (suffix after Lablink_670_training_normal)
- Debug stage logs per dataset
- Skip/resume via _DONE.ok + expected outputs check
- Save 1 global summary CSV for all datasets

Original PCA logic adapted from PCA-TTS-v1.0.py :contentReference[oaicite:1]{index=1}
"""

import os, sys, glob, time, json, traceback, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA


# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"

# Folder containing many CSVs
CSV_PATH = f"{DIR}/dataset-glucose/24/train/pp"

OUT_DIR_BASE = f"{DIR}/PCA/result/v1.1"
MODEL_DIR_BASE = f"{DIR}/PCA/model/v1.1"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5
SPLIT_SEEDS = [42, 7, 13, 21, 99]

# Heuristic weights
W_FAILRATE = 1.0
W_SPREAD = 0.10
W_STABILITY = 0.50

PARAM_GRID = {
    "n_components": [3, 4, 5, 6, 8, 10, 12],
    "contamination": [0.01, 0.02, 0.05],
}

BASE_NAME = "Lablink_670_training_normal"


# ----------------------------
def log(msg: str):
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
    if os.path.exists(done_marker_path(out_dir)):
        return True

    common = [
        os.path.join(out_dir, "gridsearch_results_heuristic.csv"),
        os.path.join(out_dir, "anomaly_results_train.csv"),
        os.path.join(out_dir, "anomaly_results_test.csv"),
        os.path.join(out_dir, "anomaly_summary_best.txt"),
    ]
    model_files = [
        os.path.join(model_dir, "pca_model_best.joblib"),
        os.path.join(model_dir, "pca_model_best.h"),
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
        raise ValueError(f"Column '{end_col}' appears before '{start_col}'. Check CSV column order.")
    return df[cols[i0:i1 + 1]].copy()


def fit_pca_model(X_train: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=RANDOM_STATE)
    pca.fit(X_train)

    mean = pca.mean_.astype(np.float32)                  # (d,)
    components = pca.components_.astype(np.float32)      # (k,d)
    eigvals = pca.explained_variance_.astype(np.float32) # (k,)

    eigvals = np.maximum(eigvals, 1e-12).astype(np.float32)
    return pca, mean, components, eigvals


def pca_scores_T2_Q(X: np.ndarray, mean: np.ndarray, components: np.ndarray, eigvals: np.ndarray):
    Xc = (X - mean).astype(np.float32)
    z = Xc @ components.T
    T2 = np.sum((z * z) / eigvals, axis=1).astype(np.float32)

    Xhat = mean + (z @ components)
    r = (X - Xhat).astype(np.float32)
    Q = np.sum(r * r, axis=1).astype(np.float32)

    T2 = np.maximum(T2, 0.0)
    Q = np.maximum(Q, 0.0)
    return T2, Q


def thresholds_from_contamination(T2_train: np.ndarray, Q_train: np.ndarray, contamination: float):
    q = 1.0 - float(contamination)
    T2_thr = float(np.quantile(T2_train, q))
    Q_thr = float(np.quantile(Q_train, q))
    return T2_thr, Q_thr


def predict_from_thresholds(T2: np.ndarray, Q: np.ndarray, T2_thr: float, Q_thr: float) -> np.ndarray:
    return ((T2 > T2_thr) | (Q > Q_thr)).astype(np.int32)


def heuristic_eval_one_split(X_np: np.ndarray, params: dict, seed: int):
    X_train, X_val = train_test_split(X_np, test_size=TEST_SIZE, random_state=seed, shuffle=True)

    _, mean, comps, eigvals = fit_pca_model(X_train, params["n_components"])
    T2_tr, Q_tr = pca_scores_T2_Q(X_train, mean, comps, eigvals)
    T2_thr, Q_thr = thresholds_from_contamination(T2_tr, Q_tr, params["contamination"])

    T2_val, Q_val = pca_scores_T2_Q(X_val, mean, comps, eigvals)
    y_val = predict_from_thresholds(T2_val, Q_val, T2_thr, Q_thr)

    val_fail_rate = float(np.mean(y_val == 1))

    comb = 0.5 * (T2_val / max(T2_thr, 1e-12) + Q_val / max(Q_thr, 1e-12))
    val_score_std = float(np.std(comb))

    return val_fail_rate, val_score_std


def export_pca_joblib_to_header(joblib_path: str, header_path: str, header_guard: str):
    bundle = joblib.load(joblib_path)

    mean = np.asarray(bundle["mean"], dtype=np.float32)
    components = np.asarray(bundle["components"], dtype=np.float32)
    eigvals = np.asarray(bundle["eigvals"], dtype=np.float32)
    T2_thr = float(bundle["T2_threshold"])
    Q_thr = float(bundle["Q_threshold"])

    d = int(mean.shape[0])
    k = int(components.shape[0])

    lines = []
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}")
    lines.append("")
    lines.append("// Auto-generated PCA anomaly detection model")
    lines.append("// Uses Hotelling T^2 + Q(SPE) statistics")
    lines.append("// fail if (T2 > PCA_T2_THRESHOLD) OR (Q > PCA_Q_THRESHOLD)")
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
    lines.append(f"#define PCA_N_FEATURES {d}")
    lines.append(f"#define PCA_N_COMPONENTS {k}")
    lines.append(f"#define PCA_T2_THRESHOLD {T2_thr:.10f}f")
    lines.append(f"#define PCA_Q_THRESHOLD  {Q_thr:.10f}f")
    lines.append("")

    lines.append(f"static const float PCA_MEAN[{d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in mean))
    lines.append("};\n")

    comp_flat = components.reshape(-1)
    lines.append(f"static const float PCA_COMPONENTS[{k*d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in comp_flat))
    lines.append("};\n")

    lines.append(f"static const float PCA_EIGVALS[{k}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in eigvals))
    lines.append("};\n")

    lines.append("static inline float pca_read_f32(const float* p, int32_t idx) {")
    lines.append("#ifdef ARDUINO")
    lines.append("  union { uint32_t u; float f; } v;")
    lines.append("  v.u = (uint32_t)pgm_read_dword((const uint32_t*)(p + idx));")
    lines.append("  return v.f;")
    lines.append("#else")
    lines.append("  return p[idx];")
    lines.append("#endif")
    lines.append("}\n")

    lines.append("static inline void pca_compute_T2_Q(const float* x, float* out_T2, float* out_Q) {")
    lines.append("  float z[PCA_N_COMPONENTS];")
    lines.append("  float xr[PCA_N_FEATURES];")
    lines.append("  for (int j=0; j<PCA_N_FEATURES; ++j) xr[j] = x[j] - pca_read_f32(PCA_MEAN, j);")
    lines.append("  for (int i=0; i<PCA_N_COMPONENTS; ++i) {")
    lines.append("    float acc = 0.0f;")
    lines.append("    int base = i * PCA_N_FEATURES;")
    lines.append("    for (int j=0; j<PCA_N_FEATURES; ++j) acc += xr[j] * pca_read_f32(PCA_COMPONENTS, base + j);")
    lines.append("    z[i] = acc;")
    lines.append("  }")
    lines.append("  float T2 = 0.0f;")
    lines.append("  for (int i=0; i<PCA_N_COMPONENTS; ++i) {")
    lines.append("    float ev = pca_read_f32(PCA_EIGVALS, i);")
    lines.append("    float zi = z[i];")
    lines.append("    T2 += (zi * zi) / (ev > 1e-12f ? ev : 1e-12f);")
    lines.append("  }")
    lines.append("  if (T2 < 0.0f) T2 = 0.0f;")
    lines.append("  float Q = 0.0f;")
    lines.append("  for (int j=0; j<PCA_N_FEATURES; ++j) {")
    lines.append("    float xhat = pca_read_f32(PCA_MEAN, j);")
    lines.append("    for (int i=0; i<PCA_N_COMPONENTS; ++i) xhat += z[i] * pca_read_f32(PCA_COMPONENTS, i*PCA_N_FEATURES + j);")
    lines.append("    float r = x[j] - xhat;")
    lines.append("    Q += r * r;")
    lines.append("  }")
    lines.append("  if (Q < 0.0f) Q = 0.0f;")
    lines.append("  *out_T2 = T2;")
    lines.append("  *out_Q = Q;")
    lines.append("}\n")

    lines.append("static inline int pca_predict(const float* x) {")
    lines.append("  float T2, Q;")
    lines.append("  pca_compute_T2_Q(x, &T2, &Q);")
    lines.append("  return (T2 > PCA_T2_THRESHOLD || Q > PCA_Q_THRESHOLD) ? 1 : 0;")
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

    # STAGE 1: load
    log("[STAGE 1/6] Loading CSV...")
    df = pd.read_csv(csv_file)
    n_rows_raw = int(len(df))

    # STAGE 2: build X + clean
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

    # STAGE 3: split
    log("[STAGE 3/6] Train/Test split...")
    X_train, X_test, meta_train, meta_test = train_test_split(
        X_np, meta_clean, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    n_train = int(len(X_train))
    n_test = int(len(X_test))
    log(f"[INFO ] Split -> train={n_train}, test={n_test}")

    # STAGE 4: grid search
    log("[STAGE 4/6] Heuristic grid search...")
    results = []
    best_obj = float("inf")
    best_params = None

    candidates = list(ParameterGrid(PARAM_GRID))
    total_candidates = len(candidates)

    for idx, params in enumerate(candidates, start=1):
        if params["n_components"] >= X_train.shape[1]:
            continue

        log(f"[GRID ] Candidate {idx}/{total_candidates}: {params}")

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

    # STAGE 5: train best model
    log("[STAGE 5/6] Training final best PCA model...")
    pca, mean, comps, eigvals = fit_pca_model(X_train, best_params["n_components"])
    T2_tr, Q_tr = pca_scores_T2_Q(X_train, mean, comps, eigvals)
    T2_thr, Q_thr = thresholds_from_contamination(T2_tr, Q_tr, best_params["contamination"])

    # STAGE 6: save outputs
    log("[STAGE 6/6] Saving model + header + results...")

    best_model_path = os.path.join(model_dir, "pca_model_best.joblib")
    joblib.dump(
        {
            "pca": pca,
            "mean": mean,
            "components": comps,
            "eigvals": eigvals,
            "T2_threshold": float(T2_thr),
            "Q_threshold": float(Q_thr),
            "spectral_columns": list(X_clean.columns),
            "config": {
                **best_params,
                "random_state": RANDOM_STATE,
                "train_size": n_train,
                "test_size": n_test,
                "grid_search_mode": "heuristic_train_only",
                "dataset_file": os.path.basename(csv_file),
                "dataset_tag": tag,
                "decision_rule": "fail_if_T2>thr_OR_Q>thr",
            },
        },
        best_model_path,
    )

    best_header_path = os.path.join(model_dir, "pca_model_best.h")
    export_pca_joblib_to_header(
        best_model_path,
        best_header_path,
        header_guard=f"PCA_MODEL_BEST_H_{tag.upper()}".replace("-", "_"),
    )

    # Train results
    y_train = predict_from_thresholds(T2_tr, Q_tr, T2_thr, Q_thr)
    comb_tr = 0.5 * (T2_tr / max(T2_thr, 1e-12) + Q_tr / max(Q_thr, 1e-12))

    train_out = pd.DataFrame({
        "Sample": meta_train["Sample"].values,
        "glucose": meta_train["glucose"].values,
        "is_anomaly": y_train,
        "status": np.where(y_train == 1, "fail", "pass"),
        "anomaly_score": comb_tr,
        "T2": T2_tr,
        "Q": Q_tr,
        "split": "train",
    })
    train_csv_path = os.path.join(out_dir, "anomaly_results_train.csv")
    train_out.to_csv(train_csv_path, index=False)

    # Test results
    T2_te, Q_te = pca_scores_T2_Q(X_test, mean, comps, eigvals)
    y_test = predict_from_thresholds(T2_te, Q_te, T2_thr, Q_thr)
    comb_te = 0.5 * (T2_te / max(T2_thr, 1e-12) + Q_te / max(Q_thr, 1e-12))

    test_out = pd.DataFrame({
        "Sample": meta_test["Sample"].values,
        "glucose": meta_test["glucose"].values,
        "is_anomaly": y_test,
        "status": np.where(y_test == 1, "fail", "pass"),
        "anomaly_score": comb_te,
        "T2": T2_te,
        "Q": Q_te,
        "split": "test",
    })
    test_csv_path = os.path.join(out_dir, "anomaly_results_test.csv")
    test_out.to_csv(test_csv_path, index=False)

    # Summary txt
    summary_path = os.path.join(out_dir, "anomaly_summary_best.txt")

    def summarize(y):
        return {
            "total": int(len(y)),
            "fail": int((y == 1).sum()),
            "pass": int((y == 0).sum()),
            "fail_rate_percent": float((y == 1).mean() * 100.0),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PCA Anomaly Detection (T² + Q) — Heuristic, Train/Test Split\n\n")
        f.write(f"Dataset file: {os.path.basename(csv_file)}\n")
        f.write(f"Dataset tag:  {tag}\n\n")
        f.write(f"Train samples: {n_train}\n")
        f.write(f"Test samples:  {n_test}\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"T2 threshold: {T2_thr:.10f}\n")
        f.write(f"Q  threshold: {Q_thr:.10f}\n\n")
        f.write(f"Train summary: {summarize(y_train)}\n")
        f.write(f"Test summary:  {summarize(y_test)}\n")

    elapsed = time.time() - t0
    log(f"[DONE ] {tag} | elapsed={elapsed:.1f}s | train_fail={(y_train==1).sum()} | test_fail={(y_test==1).sum()}")

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
        "T2_threshold": float(T2_thr),
        "Q_threshold": float(Q_thr),

        "train_fail": int((y_train == 1).sum()),
        "train_pass": int((y_train == 0).sum()),
        "train_fail_rate_percent": round(float(np.mean(y_train == 1) * 100.0), 6),

        "test_fail": int((y_test == 1).sum()),
        "test_pass": int((y_test == 0).sum()),
        "test_fail_rate_percent": round(float(np.mean(y_test == 1) * 100.0), 6),

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
                "T2_threshold": "",
                "Q_threshold": "",

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

    log("\nAll PCA datasets completed ✅")
    log(f"Global summary saved at: {summary_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
