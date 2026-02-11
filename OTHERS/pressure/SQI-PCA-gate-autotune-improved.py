# /mnt/data/SQI-PCA-gate-autotune.py
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------------- USER CONFIG ----------------
ROOT = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
CSV_PATH       = f"{ROOT}/dataset/Lablink_1k_glucose_RAW.csv"        # mixed (train)
VALID_PRESSURE = f"{ROOT}/dataset/thumb_spectral_pressure.csv"       # has labels for template
OUT_DIR        = f"{ROOT}/output/PCA-gate/pca_gate_out-improved-3"

# Hb band (your range)
BAND_LOW, BAND_HIGH = 515, 630

# PCA on CLEAN ONLY
K              = 2
THR_PCT        = 97.5

# Anchor selection (auto) & affine+projection options
ANCHOR_MODE        = "auto"   # "auto" (use template extrema) or "fixed"
ANCHOR_FIXED_NM    = [560, 585]  # used if ANCHOR_MODE="fixed" and both exist in band
EDGE_TAPER_N       = 2
BLEND_ON_DEVICE    = 1.0      # device fallback blend if needed (validator can do 1.0)
# ---------------------------------------------

def svd_pca(Xc, k):
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k, :], s[:k]

def z_recon_error(X, mu, std, Vk):
    Xz = (X - mu) / std
    Z  = Xz @ Vk.T
    Xz_hat = Z @ Vk
    R  = Xz - Xz_hat
    return np.linalg.norm(R, axis=1)

def pick_anchors_from_template(t, wl_band):
    """
    Choose 2 anchors where template carries the most structure:
    - global minimum and global maximum in the Hb band
    If they coincide (degenerate), fall back to ends.
    """
    i_min = int(np.argmin(t))
    i_max = int(np.argmax(t))
    if i_min == i_max:
        return [0, len(t)-1], [wl_band[0], wl_band[-1]]
    return [i_min, i_max], [wl_band[i_min], wl_band[i_max]]


def main():
    # ---------------- Load Lablink (mixed) ----------------
    df_train = pd.read_csv(CSV_PATH)
    wcols = [c for c in df_train.columns if "nm" in c]
    if not wcols:
        raise ValueError("No wavelength columns found (e.g., '555 nm').")
    wl_all = np.array([int(c.split()[0]) for c in wcols], dtype=int)
    X_full = df_train[wcols].to_numpy(float)

    # Hb-band slice
    band_mask = (wl_all >= BAND_LOW) & (wl_all <= BAND_HIGH)
    band_idx = np.where(band_mask)[0].tolist()
    if len(band_idx) < 2:
        raise ValueError("Hb band has < 2 channels.")
    wl_band = wl_all[band_mask]
    X_band  = X_full[:, band_mask]

    # ---------------- Build CLEAN template from labeled file ----------------
    # Use the "no pressure" rows only as the CLEAN cohort
    df_val = pd.read_csv(VALID_PRESSURE)
    wcols_val = [c for c in df_val.columns if "nm" in c]
    wl_val = np.array([int(c.split()[0]) for c in wcols_val], dtype=int)
    # Align validation columns to training order
    align_idx = [np.where(wl_val == w)[0][0] for w in wl_all]
    X_val_all = df_val[wcols_val].to_numpy(float)[:, align_idx]
    y_val = df_val["pressure"].astype(str).str.lower().map({"no":0, "yes":1}).to_numpy()

    clean = X_val_all[y_val == 0][:, band_idx]
    if clean.shape[0] < 5:
        # fallback: if very few clean samples exist, use lowest-error 20% from Lablink via quick PCA
        X = X_band
        mu0, std0 = X.mean(0), X.std(0, ddof=1) + 1e-8
        Vk0, _ = svd_pca((X - mu0)/std0, k=K)
        errs = z_recon_error(X, mu0, std0, Vk0)
        thr0 = np.quantile(errs, 0.2)
        clean = X[errs <= thr0]
    template = np.median(clean, axis=0)  # robust central clean curve

    # ---------------- PCA on CLEAN ONLY ----------------
    mu = clean.mean(0)
    std = clean.std(0, ddof=1) + 1e-8
    Vk, _ = svd_pca((clean - mu)/std, k=K)

    # threshold from clean residuals
    thr = float(np.percentile(z_recon_error(clean, mu, std, Vk), THR_PCT))

    # ---------- Label Lablink rows and optionally refit on auto-clean ----------

    LABEL_AND_OPTIONALLY_REFIT = False  # keep FALSE unless you want to refit on clean-only

    # Get sample IDs from dataframe (must exist in loaded dataset)
    sample_ids = df_train['Sample'].values  # <--- NEW: use actual sample column

    # Compute residuals of EVERY Lablink Hb-band sample against current (clean) PCA model
    errs_train = z_recon_error(X_band, mu, std, Vk)
    is_pressure = errs_train >= thr
    is_clean = ~is_pressure

    # Save labels and indices for the 1000 Lablink rows using SAMPLE column
    out_dir = Path(OUT_DIR);
    out_dir.mkdir(parents=True, exist_ok=True)

    lablink_labels = pd.DataFrame({
        "Sample": sample_ids,
        "residual": errs_train,
        "label": np.where(is_pressure, "pressure", "no_pressure")
    })
    lablink_labels.to_csv(out_dir / "lablink_auto_labels.csv", index=False)

    pd.DataFrame({
        "Sample": sample_ids[is_clean]
    }).to_csv(out_dir / "lablink_clean_index.csv", index=False)

    pd.DataFrame({
        "Sample": sample_ids[is_pressure]
    }).to_csv(out_dir / "lablink_pressure_index.csv", index=False)

    print(f"[Labeling] Lablink clean={is_clean.sum()}  pressure={is_pressure.sum()}  of total={len(X_band)}")

    # Optional: refit PCA on the auto-clean subset for a fully clean-trained model
    if LABEL_AND_OPTIONALLY_REFIT and is_clean.sum() >= max(20, 2 * K):
        Xc = X_band[is_clean]
        mu = Xc.mean(0)
        std = Xc.std(0, ddof=1) + 1e-8
        Vk, _ = svd_pca((Xc - mu) / std, k=K)
        thr = float(np.percentile(z_recon_error(Xc, mu, std, Vk), THR_PCT))
        template = np.median(Xc, axis=0)
        print(f"[Refit] Clean subset size={len(Xc)} → new THR={thr:.6f}")
    # ---------- end patch ----------

    # ---------------- Anchor selection ----------------
    if ANCHOR_MODE == "fixed":
        # choose nearest wavelengths inside band
        picks = []
        for nm in ANCHOR_FIXED_NM:
            if nm < wl_band[0] or nm > wl_band[-1]:
                continue
            picks.append(int(np.argmin(np.abs(wl_band - nm))))
        if len(picks) < 2:
            picks, _ = pick_anchors_from_template(template, wl_band.tolist())
    else:
        picks, _ = pick_anchors_from_template(template, wl_band.tolist())

    anchors_idx = picks[:2]
    anchors_nm  = [int(wl_band[i]) for i in anchors_idx]

    # ---------------- Save constants & MCU header ----------------
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    constants = {
        "band_indices": band_idx,
        "band_wavelengths": [int(x) for x in wl_band.tolist()],
        "mu": mu.tolist(),
        "std": std.tolist(),
        "vk": Vk.tolist(),
        "k": K,
        "thr": thr,
        "template": template.tolist(),
        "anchors_idx": anchors_idx,
        "anchors_nm": anchors_nm,
        "edge_taper_n": EDGE_TAPER_N,
        "device_blend": BLEND_ON_DEVICE,
        "thr_pct": THR_PCT
    }
    (out_dir/"pca_constants.json").write_text(json.dumps(constants, indent=2))

    # MCU header (adds anchor-constrained affine + PCA projection)
    header = f"""// =============================================
// AUTO-GENERATED Template-Constrained PCA (Hb band)
// =============================================
#pragma once
#include <math.h>
#include <stddef.h>
namespace PCAGate {{
static const int D  = {len(wl_band)};
static const int K  = {K};
static const float THR = {thr:.8f};
static const int EDGE_TAPER_N = {EDGE_TAPER_N};
static const float DEVICE_BLEND = {BLEND_ON_DEVICE:.6f};

static const int BAND_IDX[D] = {{ {", ".join(map(str, band_idx))} }};
static const int ANCHORS[2]  = {{ {anchors_idx[0]}, {anchors_idx[1]} }};

static const float TEMPLATE[D] = {{ {", ".join(f"{v:.8f}" for v in template)} }};
static const float MU[D] = {{ {", ".join(f"{v:.8f}" for v in mu)} }};
static const float STD[D] = {{ {", ".join(f"{v:.8f}" for v in std)} }};
static const float VK[K][D] = {{
{chr(10).join("  {" + ", ".join(f"{v:.8f}" for v in row) + "}" for row in Vk)}
}};

inline void edge_taper(float* w) {{
  for (int i=0;i<D;i++) w[i]=1.f;
  for (int t=0; t<EDGE_TAPER_N && t<D; ++t) {{
    float a = 0.5f*(1.f - cosf(3.1415926f*(t+1)/(EDGE_TAPER_N+1)));
    w[t] *= a; w[D-1-t] *= a;
  }}
}}

inline void to_z(const float* x, float* xz) {{
  for (int i=0;i<D;i++) xz[i]=(x[i]-MU[i])/STD[i];
}}
inline void from_z(const float* xz, float* x) {{
  for (int i=0;i<D;i++) x[i]=xz[i]*STD[i]+MU[i];
}}
inline void project(const float* xz, float* z) {{
  for (int k=0;k<K;k++) {{
    float acc=0.f; for (int i=0;i<D;i++) acc += xz[i]*VK[k][i];
    z[k]=acc;
  }}
}}
inline void reconstruct(const float* z, float* xz_hat) {{
  for (int i=0;i<D;i++) {{
    float acc=0.f; for (int k=0;k<K;k++) acc += z[k]*VK[k][i];
    xz_hat[i]=acc;
  }}
}}
inline float residual(const float* x) {{
  float xz[D], z[K], xzh[D];
  to_z(x, xz); project(xz,z); reconstruct(z,xzh);
  for (int i=0;i<D;i++) xz[i]-=xzh[i];
  float s=0.f; for (int i=0;i<D;i++) s+=xz[i]*xz[i];
  return sqrtf(s);
}}

// Solve a & b so that a*x + b matches TEMPLATE at two anchors.
inline void affine_match_2pt(const float* xb, float* a, float* b) {{
  int i1 = ANCHORS[0], i2 = ANCHORS[1];
  float x1 = xb[i1], x2 = xb[i2];
  float t1 = TEMPLATE[i1], t2 = TEMPLATE[i2];
  float denom = (x2 - x1);
  if (fabsf(denom) < 1e-8f) {{ *a = 1.f; *b = t1 - x1; return; }}
  *a = (t2 - t1) / denom;
  *b = t1 - (*a) * x1;
}}

// Main correction: affine to template at anchors, then PCA-project
inline void pca_fix_band(float* x_full) {{
  float xb[D]; for (int j=0;j<D;j++) xb[j] = x_full[BAND_IDX[j]];

  // 1) optional detection gate (skip if already gated upstream)
  float err0 = residual(xb);
  if (err0 < THR) return;

  // 2) two-point affine alignment to template
  float a,b; affine_match_2pt(xb, &a, &b);
  for (int i=0;i<D;i++) xb[i] = a*xb[i] + b;

  // 3) project aligned band onto clean PCA subspace
  float xz[D], z[K], xzh[D], xr[D];
  to_z(xb, xz); project(xz, z); reconstruct(z, xzh); from_z(xzh, xr);

  // 4) replace band (full blend by default), edge taper
  float w[D]; edge_taper(w);
  for (int j=0;j<D;j++) {{
    float bld = DEVICE_BLEND * w[j];
    x_full[BAND_IDX[j]] = (1.f - bld)*x_full[BAND_IDX[j]] + bld*xr[j];
  }}
}}
}} // namespace PCAGate
"""
    (out_dir/"pca_constants.json").write_text(json.dumps(constants, indent=2))
    (out_dir/"pca_gate.h").write_text(header)
    (out_dir/"pca_gate.cpp").write_text('#include "pca_gate.h"\n')

    print("Hb-band wavelengths:", wl_band.tolist())
    print("Anchors (idx,nm):", list(zip(anchors_idx, anchors_nm)))
    print(f"K={K}  THR_PCT={THR_PCT}  →  THR={thr:.6f}")
    print("Saved constants and headers into:", OUT_DIR)

if __name__ == "__main__":
    main()
