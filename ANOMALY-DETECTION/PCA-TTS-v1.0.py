#!/usr/bin/env python3
"""
PCA Anomaly Detection Training (T² + Q/SPE)

Flow matches IF/MD/SVM/LOF:
1) Load CSV, select 415..940 nm
2) Clean rows (finite)
3) One global 80/20 train-test split
4) Heuristic grid search (TRAIN ONLY)
5) Train final PCA model on TRAIN only
6) Save best model (joblib) + ESP32 header
7) Save anomaly_results_train.csv & anomaly_results_test.csv + summary
"""

import os, sys, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA


# ----------------------------
# Paths (same as IF; only IF -> PCA)
# ----------------------------
DIR = "/home/apc-3/PycharmProjects/PythonProjectAD"
CSV_PATH = f"{DIR}/dataset-glucose/raw/Lablink_670_training_normal.csv"

OUT_DIR = f"{DIR}/PCA/result/v1.0"
MODEL_DIR = f"{DIR}/PCA/model/v1.0"

RANDOM_STATE = 42

# --- Heuristic grid search controls ---
TEST_SIZE = 0.20
N_SPLITS = 5
SPLIT_SEEDS = [42, 7, 13, 21, 99]

# Heuristic weights (same style as your others)
W_FAILRATE = 1.0
W_SPREAD = 0.10
W_STABILITY = 0.50

# Grid to search
# - n_components: number of PCs kept
# - contamination: target fraction of normals flagged as fail (thresholds are quantiles)
PARAM_GRID = {
    "n_components": [3, 4, 5, 6, 8, 10, 12],
    "contamination": [0.01, 0.02, 0.05],
    # svd_solver auto is fine for this small feature set
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
        raise ValueError(f"Column '{end_col}' appears before '{start_col}'. Check CSV column order.")
    return df[cols[i0 : i1 + 1]].copy()


def fit_pca_model(X_train: np.ndarray, n_components: int):
    """
    Fit PCA on X_train (assumed float32), return (pca, mean, components, eigenvalues).
    """
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=RANDOM_STATE)
    pca.fit(X_train)

    mean = pca.mean_.astype(np.float32)                     # (d,)
    components = pca.components_.astype(np.float32)         # (k,d)
    # explained_variance_ are eigenvalues of covariance in PCA space
    eigvals = pca.explained_variance_.astype(np.float32)    # (k,)

    # Numerical safety
    eigvals = np.maximum(eigvals, 1e-12).astype(np.float32)

    return pca, mean, components, eigvals


def pca_scores_T2_Q(X: np.ndarray, mean: np.ndarray, components: np.ndarray, eigvals: np.ndarray):
    """
    Compute:
      - z = (X-mean) @ components.T
      - T2 = sum_i (z_i^2 / eigval_i)
      - X_hat = mean + z @ components
      - Q = sum_j (X - X_hat)^2
    Returns (T2, Q) as float32 arrays of shape (n,)
    """
    Xc = (X - mean).astype(np.float32)
    z = Xc @ components.T                      # (n,k)
    T2 = np.sum((z * z) / eigvals, axis=1).astype(np.float32)

    Xhat = mean + (z @ components)             # (n,d)
    r = (X - Xhat).astype(np.float32)
    Q = np.sum(r * r, axis=1).astype(np.float32)

    # clamp small negatives from numeric
    T2 = np.maximum(T2, 0.0)
    Q = np.maximum(Q, 0.0)
    return T2, Q


def thresholds_from_contamination(T2_train: np.ndarray, Q_train: np.ndarray, contamination: float):
    """
    Set thresholds on TRAIN so that approx contamination fraction fails.
    We use OR rule: fail if T2>T2_thr OR Q>Q_thr.
    To keep fail-rate near contamination, we set both thresholds at same quantile.
    This is a heuristic, but consistent and simple.
    """
    q = 1.0 - float(contamination)
    T2_thr = float(np.quantile(T2_train, q))
    Q_thr = float(np.quantile(Q_train, q))
    return T2_thr, Q_thr


def predict_from_thresholds(T2: np.ndarray, Q: np.ndarray, T2_thr: float, Q_thr: float) -> np.ndarray:
    """
    Return 0=pass(inlier), 1=fail(outlier) using OR rule.
    """
    return ((T2 > T2_thr) | (Q > Q_thr)).astype(np.int32)


def heuristic_eval_one_split(X_np: np.ndarray, params: dict, seed: int):
    """
    Fit PCA on train subset, compute thresholds from that train subset,
    evaluate fail-rate on val subset (normal-only).
    Returns: (val_fail_rate, val_score_std)
    """
    X_train, X_val = train_test_split(X_np, test_size=TEST_SIZE, random_state=seed, shuffle=True)

    _, mean, comps, eigvals = fit_pca_model(X_train, params["n_components"])
    T2_tr, Q_tr = pca_scores_T2_Q(X_train, mean, comps, eigvals)
    T2_thr, Q_thr = thresholds_from_contamination(T2_tr, Q_tr, params["contamination"])

    T2_val, Q_val = pca_scores_T2_Q(X_val, mean, comps, eigvals)
    y_val = predict_from_thresholds(T2_val, Q_val, T2_thr, Q_thr)

    val_fail_rate = float(np.mean(y_val == 1))

    # A single "score spread" proxy (higher = more anomalous):
    # normalized combined score: 0.5*(T2/T2_thr + Q/Q_thr)
    comb = 0.5 * (T2_val / max(T2_thr, 1e-12) + Q_val / max(Q_thr, 1e-12))
    val_score_std = float(np.std(comb))

    return val_fail_rate, val_score_std


def export_pca_joblib_to_header(joblib_path: str, header_path: str, header_guard: str = "PCA_MODEL_H"):
    """
    Export PCA mean, components, eigenvalues, and thresholds to C header for ESP32.
    Inference computes T2 and Q then applies OR rule.
    """
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
    lines.append("// fail if (T2 > T2_THRESHOLD) OR (Q > Q_THRESHOLD)")
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

    # mean
    lines.append(f"static const float PCA_MEAN[{d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in mean))
    lines.append("};\n")

    # components row-major (k x d)
    comp_flat = components.reshape(-1)
    lines.append(f"static const float PCA_COMPONENTS[{k*d}] PROGMEM = " + "{")
    lines.append(",".join(f"{float(x):.7g}f" for x in comp_flat))
    lines.append("};\n")

    # eigenvalues (k,)
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
    lines.append("  // Compute z = (x-mean) @ components^T")
    lines.append("  float z[PCA_N_COMPONENTS];")
    lines.append("  float xr[PCA_N_FEATURES];")
    lines.append("  for (int j=0; j<PCA_N_FEATURES; ++j) {")
    lines.append("    xr[j] = x[j] - pca_read_f32(PCA_MEAN, j);")
    lines.append("  }")
    lines.append("  for (int i=0; i<PCA_N_COMPONENTS; ++i) {")
    lines.append("    float acc = 0.0f;")
    lines.append("    int base = i * PCA_N_FEATURES;")
    lines.append("    for (int j=0; j<PCA_N_FEATURES; ++j) {")
    lines.append("      float pij = pca_read_f32(PCA_COMPONENTS, base + j);")
    lines.append("      acc += xr[j] * pij;")
    lines.append("    }")
    lines.append("    z[i] = acc;")
    lines.append("  }")
    lines.append("  // T2 = sum(z_i^2 / eig_i)")
    lines.append("  float T2 = 0.0f;")
    lines.append("  for (int i=0; i<PCA_N_COMPONENTS; ++i) {")
    lines.append("    float ev = pca_read_f32(PCA_EIGVALS, i);")
    lines.append("    float zi = z[i];")
    lines.append("    T2 += (zi * zi) / (ev > 1e-12f ? ev : 1e-12f);")
    lines.append("  }")
    lines.append("  if (T2 < 0.0f) T2 = 0.0f;")
    lines.append("  // Reconstruct x_hat = mean + z @ components")
    lines.append("  float Q = 0.0f;")
    lines.append("  for (int j=0; j<PCA_N_FEATURES; ++j) {")
    lines.append("    float xhat = pca_read_f32(PCA_MEAN, j);")
    lines.append("    for (int i=0; i<PCA_N_COMPONENTS; ++i) {")
    lines.append("      float pij = pca_read_f32(PCA_COMPONENTS, i*PCA_N_FEATURES + j);")
    lines.append("      xhat += z[i] * pij;")
    lines.append("    }")
    lines.append("    float r = x[j] - xhat;")
    lines.append("    Q += r * r;")
    lines.append("  }")
    lines.append("  if (Q < 0.0f) Q = 0.0f;")
    lines.append("  *out_T2 = T2;")
    lines.append("  *out_Q = Q;")
    lines.append("}\n")

    lines.append("static inline int pca_predict(const float* x) {")
    lines.append("  // return 0=pass, 1=fail")
    lines.append("  float T2, Q;")
    lines.append("  pca_compute_T2_Q(x, &T2, &Q);")
    lines.append("  return (T2 > PCA_T2_THRESHOLD || Q > PCA_Q_THRESHOLD) ? 1 : 0;")
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
        # Safety: cannot keep more components than features
        if params["n_components"] >= X_train.shape[1]:
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

        if best_params is None or obj < best_obj:
            best_obj = obj
            best_params = params

    results_df = pd.DataFrame(results).sort_values("objective", ascending=True)
    grid_path = os.path.join(OUT_DIR, "gridsearch_results_heuristic.csv")
    results_df.to_csv(grid_path, index=False)

    print("\n[GridSearch] Best params:", best_params)
    print(f"[GridSearch] Saved grid results: {grid_path}")

    # 5) Train final PCA model on TRAIN only
    pca, mean, comps, eigvals = fit_pca_model(X_train, best_params["n_components"])
    T2_tr, Q_tr = pca_scores_T2_Q(X_train, mean, comps, eigvals)
    T2_thr, Q_thr = thresholds_from_contamination(T2_tr, Q_tr, best_params["contamination"])

    # 6) Save joblib
    best_model_path = os.path.join(MODEL_DIR, "pca_model_best.joblib")
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
                "train_size": len(X_train),
                "test_size": len(X_test),
                "grid_search_mode": "heuristic_train_only",
                "decision_rule": "fail_if_T2>thr_OR_Q>thr",
            },
        },
        best_model_path,
    )

    # 7) Export header
    best_header_path = os.path.join(MODEL_DIR, "pca_model_best.h")
    export_pca_joblib_to_header(best_model_path, best_header_path, header_guard="PCA_MODEL_BEST_H")

    # 8) Train outputs
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
    train_out.to_csv(os.path.join(OUT_DIR, "anomaly_results_train.csv"), index=False)

    # 9) Test outputs
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
    test_out.to_csv(os.path.join(OUT_DIR, "anomaly_results_test.csv"), index=False)

    # 10) Summary
    def summarize(y):
        return {
            "total": int(len(y)),
            "fail": int((y == 1).sum()),
            "pass": int((y == 0).sum()),
            "fail_rate_percent": float((y == 1).mean() * 100.0),
        }

    summary_path = os.path.join(OUT_DIR, "anomaly_summary_best.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PCA Anomaly Detection (T² + Q) — Heuristic, Train/Test Split\n\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Output Dir: {OUT_DIR}\n")
        f.write(f"Model Dir: {MODEL_DIR}\n\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples:  {len(X_test)}\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"T2 threshold: {T2_thr:.10f}\n")
        f.write(f"Q  threshold: {Q_thr:.10f}\n\n")
        f.write(f"Train summary: {summarize(y_train)}\n")
        f.write(f"Test summary:  {summarize(y_test)}\n")

    print("\nDone ✅")
    print(f"Saved BEST model:  {best_model_path}")
    print(f"Saved BEST header: {best_header_path}")
    print("Train results → anomaly_results_train.csv")
    print("Test results  → anomaly_results_test.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
