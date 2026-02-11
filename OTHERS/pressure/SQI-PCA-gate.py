#!/usr/bin/env python3
"""
make_pca_gate_hb_band.py
Train PCA gate on Hb-band (540–600 nm) from your RAW no-pressure dataset
and export ESP32-ready pca_gate.h/.cpp

Just run:
    python make_pca_gate_hb_band.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ---------------- CONFIG ----------------
ROOT = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
CSV_PATH   = f"{ROOT}/dataset/Lablink_1k_glucose_RAW.csv"  # your dataset
OUT_DIR    = f"{ROOT}/output/PCA-gate/pca_gate_out-1"                        # output folder
K          = 2        # number of PCs
THR_PCT    = 90     # percentile threshold
BLEND      = 1.0      # 1.0 = full replace
WRITE_JSON = True
# ----------------------------------------

def svd_pca(Xc, k):
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k, :]
    return Vk, s[:k]

def recon_error(X, mu, Vk):
    Xc = X - mu
    Z  = Xc @ Vk.T
    Xr = Z @ Vk
    E  = Xc - Xr
    return np.linalg.norm(E, axis=1)

def main():
    df = pd.read_csv(CSV_PATH)
    wcols = [c for c in df.columns if "nm" in c]
    wl = np.array([int(c.split()[0]) for c in wcols], dtype=int)

    # Select Hb-band (540–600 nm)
    band_mask = (wl >= 540) & (wl <= 610)
    band_idx = np.where(band_mask)[0].tolist()
    band_wl = wl[band_mask].tolist()
    X_band = df[wcols].to_numpy(float)[:, band_mask]
    N, D = X_band.shape

    # Split 80/20
    rng = np.random.default_rng(42)
    idx = np.arange(N); rng.shuffle(idx)
    split = int(0.8 * N)
    X_tr, X_te = X_band[idx[:split]], X_band[idx[split:]]

    # PCA
    mu = X_tr.mean(axis=0)
    Vk, _ = svd_pca(X_tr - mu, k=K)

    # Threshold
    err_te = recon_error(X_te, mu, Vk)
    thr = float(np.percentile(err_te, THR_PCT))

    # Optional Hb-trio indices
    def find_idx(val):
        pos = np.where(wl == val)[0]
        return int(pos[0]) if len(pos) else -1
    idx_555, idx_560, idx_585 = find_idx(555), find_idx(560), find_idx(585)

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    header = f"""#pragma once
#include <stddef.h>
#include <math.h>

namespace PCAGate {{

// Hb-band wavelengths: {band_wl}
static constexpr int D = {D};
static constexpr int BAND_IDX[D] = {{ {", ".join(map(str, band_idx))} }};

static constexpr int K = {K};
static constexpr float THR = {thr:.6f}f;
static constexpr float BLEND = {BLEND:.2f}f;

static constexpr float MU[D] = {{
  {", ".join(f"{v:.8f}f" for v in mu)}
}};

static constexpr float VK[K][D] = {{
{chr(10).join("  {" + ", ".join(f"{v:.8f}f" for v in row) + "}" for row in Vk)}
}};

// Optional Hb-trio indices in full spectrum
static constexpr int IDX_555 = {idx_555};
static constexpr int IDX_560 = {idx_560};
static constexpr int IDX_585 = {idx_585};

// --- Helpers ---
inline void gather_band(const float* x_full, float* xb) {{
    for (int i=0;i<D;i++) xb[i] = x_full[BAND_IDX[i]];
}}
inline float l2(const float* v, int n) {{
    float s=0.f; for (int i=0;i<n;i++) s+=v[i]*v[i]; return sqrtf(s);
}}
inline void project(const float* xc, float* z) {{
    for (int k=0;k<K;k++) {{
        float acc=0.f;
        for (int i=0;i<D;i++) acc += xc[i]*VK[k][i];
        z[k]=acc;
    }}
}}
inline void reconstruct(const float* z, float* xr) {{
    for (int i=0;i<D;i++) {{
        float acc=0.f;
        for (int k=0;k<K;k++) acc += z[k]*VK[k][i];
        xr[i]=acc;
    }}
}}

// --- API ---
inline bool is_pressured(const float* x_full) {{
    float xb[D]; gather_band(x_full, xb);
    for (int i=0;i<D;i++) xb[i]-=MU[i];
    float z[K]; project(xb, z);
    float xr[D]; reconstruct(z, xr);
    for (int i=0;i<D;i++) xr[i]=xb[i]-xr[i];
    const float err=l2(xr,D);
    return err>=THR;
}}

inline void pca_fix_band(float* x_full) {{
    float xb[D]; gather_band(x_full, xb);
    for (int i=0;i<D;i++) xb[i]-=MU[i];
    float z[K]; project(xb,z);
    float xr[D]; reconstruct(z,xr);
    for (int i=0;i<D;i++) {{
        xr[i]+=MU[i];
        const int j=BAND_IDX[i];
        x_full[j]=(1.0f-BLEND)*x_full[j]+BLEND*xr[i];
    }}
}}

}} // namespace PCAGate
"""

    cpp = '#include "pca_gate.h"\n'

    (out_dir/"pca_gate.h").write_text(header)
    (out_dir/"pca_gate.cpp").write_text(cpp)

    if WRITE_JSON:
        (out_dir/"pca_constants.json").write_text(json.dumps({
            "csv": CSV_PATH,
            "band_wavelengths": band_wl,
            "band_indices": band_idx,
            "k": K,
            "thr_percentile": THR_PCT,
            "thr": thr,
            "mu": mu.tolist(),
            "Vk": Vk.tolist(),
            "blend": BLEND
        }, indent=2))

    print("Hb-band wavelengths:", band_wl)
    print("Threshold:", thr)
    print("Wrote:", out_dir/"pca_gate.h")

if __name__ == "__main__":
    main()
