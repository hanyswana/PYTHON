#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# === paths (edit if needed)
OUTDIR = "/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose/model-NEW-FLOW/10th_r2_0.61_77_/xray_out"
XRAY_SIDECSV = os.path.join(OUTDIR, "XRAY_TFLite_early10_vs_external.csv")

# Expected raw column order
WCOLS = ["555 nm","560 nm","585 nm","630 nm","645 nm","680 nm","810 nm","860 nm","900 nm","940 nm"]

def norm_euclidean(X):
    # L2 normalize each row; avoid divide by zero
    X = np.asarray(X, dtype=np.float32)
    l2 = np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-8
    return X / l2

def snv(X):
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1e-8, sd)
    return (X - mu) / sd

def savgol_deriv(X, window=9, poly=2, deriv=1):
    X = np.asarray(X, dtype=np.float32)
    # apply along features axis=1
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv, axis=1)

def baseline_remove(X):
    X = np.asarray(X, dtype=np.float32)
    row_mean = X.mean(axis=1, keepdims=True)
    return X - row_mean

def external_pipeline_from_raw(raw10):
    # Pipeline: Norm_Euc → SNV → SavGol (w=9,p=2,deriv=1) → Baseline
    X1 = norm_euclidean(raw10)
    X2 = snv(X1)
    X3 = savgol_deriv(X2, window=9, poly=2, deriv=1)
    X4 = baseline_remove(X3)
    return X4.astype(np.float32)

def corr_cols(A, B):
    # column-wise correlation, then mean across columns
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)
    B = (B - B.mean(axis=0)) / (B.std(axis=0) + 1e-12)
    c = np.mean(np.sum(A*B, axis=0) / A.shape[0])
    return float(c)

def mae(A, B):
    return float(np.mean(np.abs(np.asarray(A) - np.asarray(B))))

def main():
    if not os.path.isfile(XRAY_SIDECSV):
        print(f"XRAY side-by-side CSV not found: {XRAY_SIDECSV}")
        sys.exit(1)

    df = pd.read_csv(XRAY_SIDECSV)
    # Extract early_*, external_* matrices
    early_cols    = [c for c in df.columns if c.startswith("early_")]
    external_cols = [c for c in df.columns if c.startswith("external_")]

    if len(early_cols) != 10 or len(external_cols) != 10:
        print(f"Unexpected columns. early:{len(early_cols)} external:{len(external_cols)}")
        print("Make sure you ran XRAY with a 10-band CSV (RAW) and the script dumped early vs external.")
        sys.exit(1)

    EARLY   = df[early_cols].to_numpy(dtype=np.float32)      # internal transform
    EXTERNAL_RAW = df[external_cols].to_numpy(dtype=np.float32)  # raw inputs

    # Recompute your external pipeline on these RAW inputs
    EXTERNAL_PIPE = external_pipeline_from_raw(EXTERNAL_RAW)

    # Compare
    c_match   = corr_cols(EARLY, EXTERNAL_PIPE)
    mae_match = mae(EARLY, EXTERNAL_PIPE)

    c_raw_e   = corr_cols(EARLY, EXTERNAL_RAW)
    c_raw_ext = corr_cols(EXTERNAL_PIPE, EXTERNAL_RAW)

    print("\n=== XRAY Verification ===")
    print("Rows compared:", EARLY.shape[0])
    print("Corr( EARLY , ExternalPipeline ) =", round(c_match, 6))
    print("MAE ( EARLY , ExternalPipeline ) =", round(mae_match, 6))
    print("Corr( EARLY , RAW )              =", round(c_raw_e, 6))
    print("Corr( ExternalPipeline , RAW )   =", round(c_raw_ext, 6))

    # Verdict: strong match if early ~= your pipeline
    strong_corr = c_match >= 0.95 and mae_match < 1e-3  # MAE threshold is tight since both are scaled
    if strong_corr:
        print("\nVERDICT: MATCH ✅  (Model’s internal transform ≈ your external pipeline)")
        print("→ Feed RAW (no external preprocess) to avoid double-processing.")
    else:
        print("\nVERDICT: NO MATCH ❌  (Internal transform ≠ your external pipeline)")
        print("→ Do NOT apply your external SNV/Deriv/Baseline before inference for this export.")
        print("→ This model has an internal Normalization layer that differs from your SNV pipeline.")
        print("   Validate with RAW 10-band float32 in the exact wavelength order only.")

if __name__ == "__main__":
    main()
