import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG (edit paths if needed) ----------------
ROOT         = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
CONST_FILE   = f"{ROOT}/output/PCA-gate/pca_gate_out-2/pca_constants.json"
PRESSURE_CSV = f"{ROOT}/dataset/thumb_spectral_pressure.csv"
DO_SWEEP     = True
SWEEP_PCTS   = [90, 92.5, 95, 97.5]
# ----------------------------------------------------------------

# ---------- PCA helpers (z-space) ----------
def to_z(xb, mu, std):
    return (xb - mu) / std

def from_z(xz, mu, std):
    return xz * std + mu

def project(xz, Vk):
    return xz @ Vk.T

def reconstruct(z, Vk):
    return z @ Vk

def zspace_residual_norm(xb, mu, std, Vk):
    """
    Error in z-space:
      xz = (xb - mu)/std
      xz_hat = (xz @ Vk.T) @ Vk
      err = ||xz - xz_hat||_2
    """
    xz = to_z(xb, mu, std)
    z  = project(xz, Vk)
    xz_hat = reconstruct(z, Vk)
    r  = xz - xz_hat
    if r.ndim == 1:
        return float(np.linalg.norm(r))
    return np.linalg.norm(r, axis=1)

def pca_fix_band_zspace(x_full, mu, std, Vk, band_idx, blend=1.0):
    """
    If a spectrum is flagged, reconstruct Hb-band in z-space and blend back in raw units.
    """
    xb = x_full[band_idx]
    xz = to_z(xb, mu, std)
    z  = project(xz, Vk)
    xz_hat = reconstruct(z, Vk)
    xr = from_z(xz_hat, mu, std)  # back to raw units

    x_fixed = x_full.copy()
    for i, bi in enumerate(band_idx):
        x_fixed[bi] = (1.0 - blend) * x_full[bi] + blend * xr[i]
    return x_fixed

# ---------- Metrics ----------
def confusion(flags, y):
    TP = int(((y==1) &  flags).sum())
    FN = int(((y==1) & ~flags).sum())
    TN = int(((y==0) & ~flags).sum())
    FP = int(((y==0) &  flags).sum())
    sens = TP / (TP+FN) if (TP+FN)>0 else 0.0
    spec = TN / (TN+FP) if (TN+FP)>0 else 0.0
    acc  = (TP+TN) / len(y) if len(y)>0 else 0.0
    return TP,FN,TN,FP,sens,spec,acc

def main():
    # ---- Load constants from z-space trainer ----
    cfg = json.loads(Path(CONST_FILE).read_text())
    band_idx = cfg["band_indices"]
    band_wl  = cfg["band_wavelengths"]
    mu  = np.array(cfg["mu"],  dtype=float)
    std = np.array(cfg["std"], dtype=float)
    Vk  = np.array(cfg["Vk"],  dtype=float)
    thr = float(cfg["thr"])
    blend = float(cfg.get("blend", 1.0))

    print("Hb-band wavelengths used:", band_wl)
    print("Detection threshold (z-space):", thr)

    # ---- Load labeled validation data ----
    df = pd.read_csv(PRESSURE_CSV)
    wcols = [c for c in df.columns if "nm" in c]
    wl_full = np.array([int(c.split()[0]) for c in wcols], dtype=int)
    X = df[wcols].to_numpy(float)
    y = df["pressure"].astype(str).str.lower().map({"no":0,"yes":1}).to_numpy()

    # ---- Compute z-space residual errors & apply training threshold ----
    errs = np.array([zspace_residual_norm(x[band_idx], mu, std, Vk) for x in X])
    flags = errs >= thr

    TP,FN,TN,FP,sens,spec,acc = confusion(flags, y)
    print("\nDetection @ training threshold:")
    print(f"TP={TP}  FN={FN}  TN={TN}  FP={FP}")
    print(f"Sensitivity={sens:.3f}  Specificity={spec:.3f}  Accuracy={acc:.3f}")

    # ---- Optional: sweep thresholds using NO-PRESSURE errors from validation set ----
    if DO_SWEEP:
        errs_no = errs[y==0]
        if len(errs_no) > 5:
            sweep_thrs = np.percentile(errs_no, SWEEP_PCTS)
            print("\nThreshold sweep (percentiles of NO-PRESSURE in validation set):")
            for p, t in zip(SWEEP_PCTS, sweep_thrs):
                f = errs >= t
                _TP,_FN,_TN,_FP,_sens,_spec,_acc = confusion(f, y)
                print(f"  pct={p:>5} thr={t:.6f}  TP={_TP:2d} FN={_FN:2d} TN={_TN:2d} FP={_FP:2d}  "
                      f"Sens={_sens:.2f} Spec={_spec:.2f} Acc={_acc:.2f}")
        else:
            print("\n[SWEEP] Not enough no-pressure rows to compute percentiles.")

    # ---- Apply fix to flagged rows (z-space reconstruction, then un-zscore) ----
    X_fixed = np.array([
        pca_fix_band_zspace(x, mu, std, Vk, band_idx, blend) if f else x
        for x, f in zip(X, flags)
    ])

    # ---- Plot mean Hb-band curves ----
    band_wl_arr = wl_full[band_idx]
    mean_no  = X[y==0][:, band_idx].mean(axis=0)
    mean_yes = X[y==1][:, band_idx].mean(axis=0)
    mean_fix = X_fixed[y==1][:, band_idx].mean(axis=0)

    plt.figure()
    plt.plot(band_wl_arr, mean_no,  'g-o', label="No Pressure")
    plt.plot(band_wl_arr, mean_yes, 'r-o', label="With Pressure (raw)")
    plt.plot(band_wl_arr, mean_fix, 'b-o', label="With Pressure (fixed)")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Absorbance (a.u.)"); plt.grid(True)
    plt.title("Hb-band correction (PCA gate in z-space)")
    plt.legend(); plt.show()

    # ---- Plot error histogram ----
    plt.figure()
    plt.hist([errs[y==0], errs[y==1]], bins=20,
             label=["No Pressure","With Pressure"], alpha=0.7)
    plt.axvline(thr, color='k', linestyle='--', label="Training THR (z-space)")
    plt.xlabel("Reconstruction error (z-space)"); plt.ylabel("Count")
    plt.title("Detection separation (PCA gate in z-space)")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()

