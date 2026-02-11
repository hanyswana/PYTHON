import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# ---- Load PCA constants trained on 1k no-pressure ----
ROOT = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
OUT_DIR    = f"{ROOT}/output/PCA-gate/pca_gate_out-1"                        # output folder
with open(f"{OUT_DIR}/pca_constants.json") as f:
    cfg = json.load(f)

band_idx = cfg["band_indices"]
mu = np.array(cfg["mu"])
Vk = np.array(cfg["Vk"])
thr = cfg["thr"]
wl = cfg["band_wavelengths"]
blend = cfg["blend"]

print("Hb-band wavelengths used:", wl)
print("Detection threshold:", thr)

# ---- Helper funcs ----
def project(xc, Vk):
    return xc @ Vk.T

def reconstruct(z, Vk):
    return z @ Vk

def recon_error(x, mu, Vk):
    xc = x - mu
    z = project(xc, Vk)
    xr = reconstruct(z, Vk)
    return np.linalg.norm(xc - xr)

def pca_fix_band(x_full, mu, Vk, band_idx, blend=1.0):
    xb = x_full[band_idx]
    xc = xb - mu
    z = project(xc, Vk)
    xr = reconstruct(z, Vk) + mu
    # blend back into spectrum
    x_fixed = x_full.copy()
    for i, bi in enumerate(band_idx):
        x_fixed[bi] = (1-blend)*x_full[bi] + blend*xr[i]
    return x_fixed

# ---- Load pressure dataset ----
df = pd.read_csv(f"{ROOT}/dataset/thumb_spectral_pressure.csv")
wcols = [c for c in df.columns if "nm" in c]
X = df[wcols].to_numpy(float)

labels = df["pressure"].str.lower().map({"no":0,"yes":1}).to_numpy()

# ---- Detection ----
errs = [recon_error(x[band_idx], mu, Vk) for x in X]
flags = np.array(errs) >= thr

print("Detection accuracy:")
print("True Positives (Yes detected):", np.sum((labels==1) & flags))
print("False Negatives (Yes missed):", np.sum((labels==1) & ~flags))
print("True Negatives (No clean):", np.sum((labels==0) & ~flags))
print("False Positives (No flagged):", np.sum((labels==0) & flags))

# ---- Apply fix ----
X_fixed = np.array([
    pca_fix_band(x, mu, Vk, band_idx, blend=1.0) if f else x
    for x,f in zip(X, flags)
])

# ---- Visualization ----
# Plot mean Hb-band spectra (with, no, fixed)
wl_full = [int(c.split()[0]) for c in wcols]
band_wl = np.array(wl_full)[band_idx]

mean_no = X[labels==0][:, band_idx].mean(axis=0)
mean_yes = X[labels==1][:, band_idx].mean(axis=0)
mean_fix = X_fixed[labels==1][:, band_idx].mean(axis=0)

plt.figure()
plt.plot(band_wl, mean_no, 'g-o', label="No Pressure")
plt.plot(band_wl, mean_yes, 'r-o', label="With Pressure (raw)")
plt.plot(band_wl, mean_fix, 'b-o', label="With Pressure (fixed)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (a.u.)")
plt.title("Hb-band correction with PCA gate")
plt.legend()
plt.grid(True)
plt.show()

# ---- Error distribution ----
plt.figure()
plt.hist([np.array(errs)[labels==0], np.array(errs)[labels==1]],
         bins=20, label=["No Pressure","With Pressure"], alpha=0.7)
plt.axvline(thr, color='k', linestyle='--', label="Threshold")
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.legend()
plt.title("Detection separation")
plt.show()
