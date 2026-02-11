import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------------- USER CONFIG ----------------
ROOT = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
CSV_PATH   = f"{ROOT}/dataset/Lablink_1k_glucose_RAW.csv"  # your dataset
OUT_DIR    = f"{ROOT}/output/PCA-gate/pca_gate_out-2"                        # output folder

# Hb-band window (kept within your 19 wavelengths)
BAND_LOW, BAND_HIGH = 515, 630

# PCA / threshold
K        = 2        # number of principal components
THR_PCT  = 95.0     # percentile on NO-PRESSURE holdout (z-space residual)
BLEND    = 1.0      # max fix strength used on device
SEED     = 42       # RNG seed for 80/20 split
# ---------------------------------------------

def svd_pca(Xc, k):
    """Return top-k right singular vectors (principal axes)."""
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k, :], s[:k]

def z_recon_error(X, mu, std, Vk):
    """
    Residual norm in z-space:
      Xz = (X - mu)/std
      Xz_hat = (Xz @ Vk.T) @ Vk
      err = ||Xz - Xz_hat||_2
    """
    Xz = (X - mu) / std
    Z  = Xz @ Vk.T
    Xz_hat = Z @ Vk
    R  = Xz - Xz_hat
    return np.linalg.norm(R, axis=1)

def main():
    # ---- Load NO-PRESSURE training data ----
    df = pd.read_csv(CSV_PATH)
    wcols = [c for c in df.columns if "nm" in c]
    if not wcols:
        raise ValueError("No wavelength columns found (e.g., '555 nm').")
    wl = np.array([int(c.split()[0]) for c in wcols], dtype=int)
    X_full = df[wcols].to_numpy(float)

    # ---- Select Hb-band columns that exist in your data ----
    band_mask = (wl >= BAND_LOW) & (wl <= BAND_HIGH)
    band_idx = np.where(band_mask)[0].tolist()
    if len(band_idx) < 2:
        raise ValueError("Fewer than 2 channels found in the selected Hb-band.")
    band_wl = wl[band_mask].tolist()

    X = X_full[:, band_mask]  # (N, D) Hb-band only
    N, D = X.shape

    # ---- 80/20 split (NO-PRESSURE only) ----
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    X_tr, X_ho = X[idx[:split]], X[idx[split:]]

    # ---- Fit PCA in z-space ----
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0, ddof=1) + 1e-8  # avoid div-by-zero
    Xz_tr = (X_tr - mu) / std
    Vk, _ = svd_pca(Xz_tr, k=K)

    # ---- Calibrate threshold (NO-PRESSURE holdout, z-space residual) ----
    err_ho = z_recon_error(X_ho, mu, std, Vk)
    thr = float(np.percentile(err_ho, THR_PCT))

    # ---- Write outputs ----
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # JSON for offline validation / auditing
    constants = {
        "band_wavelengths": [int(x) for x in band_wl],
        "band_indices": band_idx,
        "k": K,
        "thr_percentile": THR_PCT,
        "thr": thr,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "Vk": Vk.tolist(),
        "blend": BLEND
    }
    (out_dir/"pca_constants.json").write_text(json.dumps(constants, indent=2))

    # C++ header (ESP32)
    header = f"""#pragma once
#include <stddef.h>
#include <math.h>

namespace PCAGate {{

// Hb-band design
static constexpr int D = {D};
static constexpr int BAND_IDX[D] = {{ {", ".join(map(str, band_idx))} }};

// PCA design
static constexpr int   K    = {K};
static constexpr float THR  = {thr:.6f}f;   // threshold in z-space residual
static constexpr float BLEND = {BLEND:.2f}f;

// Standardization (per-wavelength)
static constexpr float MU[D] = {{
  {", ".join(f"{v:.8f}f" for v in mu)}
}};
static constexpr float STD[D] = {{
  {", ".join(f"{v:.8f}f" for v in std)}
}};

// Top-K principal axes (in z-space)
static constexpr float VK[K][D] = {{
{chr(10).join("  {" + ", ".join(f"{v:.8f}f" for v in row) + "}" for row in Vk)}
}};

// ---- helpers ----
inline void gather_band(const float* x_full, float* xb) {{
  #pragma unroll
  for (int i=0;i<D;i++) xb[i] = x_full[BAND_IDX[i]];
}}
inline float l2(const float* v, int n) {{
  float s=0.f; for (int i=0;i<n;i++) s += v[i]*v[i]; return sqrtf(s);
}}
inline void to_z(const float* x, float* xz) {{
  #pragma unroll
  for (int i=0;i<D;i++) xz[i] = (x[i] - MU[i]) / STD[i];
}}
inline void from_z(const float* xz, float* x) {{
  #pragma unroll
  for (int i=0;i<D;i++) x[i] = xz[i]*STD[i] + MU[i];
}}
inline void project(const float* xz, float* z) {{
  // z = xz @ VK^T
  for (int k=0;k<K;k++) {{
    float acc=0.f;
    for (int i=0;i<D;i++) acc += xz[i]*VK[k][i];
    z[k]=acc;
  }}
}}
inline void reconstruct(const float* z, float* xz_hat) {{
  // xz_hat = z @ VK
  for (int i=0;i<D;i++) {{
    float acc=0.f;
    for (int k=0;k<K;k++) acc += z[k]*VK[k][i];
    xz_hat[i]=acc;
  }}
}}

// ---- API ----
// Detect pressure: compute z-space residual on Hb-band and compare to THR
inline bool is_pressured(const float* x_full) {{
  float xb[D];    gather_band(x_full, xb);   // Hb-band (raw)
  float xz[D];    to_z(xb, xz);              // z-score band
  float z[K];     project(xz, z);            // project
  float xz_hat[D];reconstruct(z, xz_hat);    // reconstruct in z-space
  float r[D];     for(int i=0;i<D;i++) r[i] = xz[i] - xz_hat[i]; // residual (z)
  const float err = l2(r, D);
  return err >= THR;
}}

// Fix pressure: replace Hb-band with its PCA reconstruction (un-zscored)
inline void pca_fix_band(float* x_full) {{
  // 1) z-score the current Hb-band
  float xb[D]; gather_band(x_full, xb);
  float xz[D]; to_z(xb, xz);

  // 2) reconstruct in z-space
  float z[K];      project(xz, z);
  float xz_hat[D]; reconstruct(z, xz_hat);

  // 3) back to raw units
  float xr[D]; from_z(xz_hat, xr);

  // 4) blend reconstructed band back into full spectrum
  for (int i=0;i<D;i++) {{
    const int j = BAND_IDX[i];
    x_full[j] = (1.0f - BLEND)*x_full[j] + BLEND*xr[i];
  }}
}}

}} // namespace PCAGate
"""
    (out_dir/"pca_gate.h").write_text(header)
    (out_dir/"pca_gate.cpp").write_text('#include "pca_gate.h"\n')

    print("Hb-band wavelengths:", band_wl)
    print(f"Selected K={K}, THR_PCT={THR_PCT} → THR(z-space)={thr:.6f}")
    print("Wrote:", out_dir/"pca_constants.json")
    print("Wrote:", out_dir/"pca_gate.h")
    print("Wrote:", out_dir/"pca_gate.cpp")

if __name__ == "__main__":
    main()
