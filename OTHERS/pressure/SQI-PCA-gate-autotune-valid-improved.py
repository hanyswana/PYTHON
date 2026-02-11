# /mnt/data/SQI-PCA-gate-autotune-valid.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT         = "/home/apc-3/PycharmProjects/PythonProjectAK/SQI"
OUT_DIR      = f"{ROOT}/output/PCA-gate/pca_gate_out-improved-3"
CONST_FILE   = f"{OUT_DIR}/pca_constants.json"
PRESSURE_CSV = f"{ROOT}/dataset/thumb_spectral_pressure.csv"

# For plots / evaluation
DO_SWEEP     = True
SWEEP_PCTS   = [90, 92.5, 95, 97.5, 99]

# ---------- helpers (mirror header) ----------
def to_z(xb, mu, std): return (xb - mu)/std
def project(xz, Vk):    return xz @ Vk.T
def reconstruct(z, Vk): return z @ Vk
def from_z(xz, mu, std): return xz*std + mu

def residual(xb, mu, std, Vk):
    xz = to_z(xb, mu, std); z = project(xz, Vk); xzh = reconstruct(z, Vk)
    r = xz - xzh
    return float(np.linalg.norm(r))

def edge_taper_weights(D, n):
    w = np.ones(D, dtype=float)
    n = max(0, min(n, D//2))
    if n > 0:
        ramp = 0.5*(1 - np.cos(np.pi*np.linspace(1, n, n)/(n+1)))
        w[:n] *= ramp; w[-n:] *= ramp[::-1]
    return w

def affine_match_2pt(xb, template, anchors_idx):
    i1, i2 = anchors_idx
    x1, x2 = xb[i1], xb[i2]
    t1, t2 = template[i1], template[i2]
    denom = (x2 - x1)
    if abs(denom) < 1e-8:
        a = 1.0; b = t1 - x1
    else:
        a = (t2 - t1)/denom
        b = t1 - a*x1
    return a, b

def fix_one(x_full, band_idx, mu, std, Vk, thr, template, anchors_idx, edge_n, blend=1.0):
    xb = x_full[band_idx]
    if residual(xb, mu, std, Vk) < thr:
        return x_full.copy(), False
    a,b = affine_match_2pt(xb, template, anchors_idx)
    xb2 = a*xb + b
    xz = to_z(xb2, mu, std); z = project(xz, Vk); xzh = reconstruct(z, Vk)
    xr = from_z(xzh, mu, std)
    x_out = x_full.copy()
    w = edge_taper_weights(len(band_idx), edge_n)
    for i, bi in enumerate(band_idx):
        x_out[bi] = (1.0 - blend*w[i]) * x_full[bi] + (blend*w[i]) * xr[i]
    return x_out, True

def confusion(flags, y):
    TP = int(((y==1) &  flags).sum()); FN = int(((y==1) & ~flags).sum())
    TN = int(((y==0) & ~flags).sum()); FP = int(((y==0) &  flags).sum())
    sens = TP / (TP + FN + 1e-9); spec = TN / (TN + FP + 1e-9)
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    return TP, FN, TN, FP, sens, spec, acc

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load constants ----
    const = json.loads(Path(CONST_FILE).read_text())
    band_idx = const["band_indices"]
    band_wl  = np.array(const["band_wavelengths"], int)
    mu   = np.array(const["mu"], float)
    std  = np.array(const["std"], float)
    Vk   = np.array(const["vk"], float)
    thr  = float(const["thr"])
    template = np.array(const["template"], float)
    anchors_idx = const["anchors_idx"]
    edge_n = int(const["edge_taper_n"])
    device_blend = float(const.get("device_blend", 1.0))

    print("Hb-band wavelengths:", band_wl.tolist())
    print("Anchors (idx):", anchors_idx, "→ nm:", [int(band_wl[i]) for i in anchors_idx])
    print("Training threshold:", thr)

    # ---- Load labeled validation data ----
    df = pd.read_csv(PRESSURE_CSV)
    wcols = [c for c in df.columns if "nm" in c]
    wl_val = np.array([int(c.split()[0]) for c in wcols], dtype=int)
    X_all = df[wcols].to_numpy(float)
    # align to band order
    band_order_idx = [np.where(wl_val==w)[0][0] for w in band_wl]
    X = X_all[:, band_order_idx]
    y = df["pressure"].astype(str).str.lower().map({"no":0,"yes":1}).to_numpy()

    # ---- Residuals & detection ----
    errs = np.array([residual(x, mu, std, Vk) for x in X])
    flags = errs >= thr
    TP,FN,TN,FP,sens,spec,acc = confusion(flags, y)
    print(f"\nDetect @ training THR -> Sens={sens:.2f} Spec={spec:.2f} Acc={acc:.2f}")

    # ---- Threshold sweep (optional) ----
    sweep_rows = []
    if DO_SWEEP and (y==0).sum() > 5:
        errs_no = errs[y==0]
        thrs = np.percentile(errs_no, SWEEP_PCTS)
        for p,t in zip(SWEEP_PCTS, thrs):
            f = errs >= t
            _TP,_FN,_TN,_FP,_s,_c,_a = confusion(f, y)
            sweep_rows.append({
                "percentile_from_no_pressure": p,
                "threshold": float(t),
                "TP": _TP, "FN": _FN, "TN": _TN, "FP": _FP,
                "sensitivity": _s, "specificity": _c, "accuracy": _a
            })
    # append the training threshold row too
    sweep_rows.append({
        "percentile_from_no_pressure": const.get("thr_pct", 97.5),
        "threshold": float(thr),
        "TP": TP, "FN": FN, "TN": TN, "FP": FP,
        "sensitivity": sens, "specificity": spec, "accuracy": acc
    })
    pd.DataFrame(sweep_rows).to_csv(out_dir/"detection_scores.csv", index=False)

    # ---- Demonstration correction on class means (saved to CSV) ----
    X_no   = X[y==0]
    X_yes  = X[y==1]
    mean_no  = X_no.mean(0) if len(X_no) else X.mean(0)
    mean_raw = X_yes.mean(0) if len(X_yes) else X.mean(0)
    mean_fix, _ = fix_one(mean_raw.copy(), list(range(len(band_wl))),
                          mu, std, Vk, thr, template, anchors_idx, edge_n,
                          blend=device_blend)

    pd.DataFrame({
        "wavelength_nm": band_wl,
        "mean_no_pressure": mean_no,
        "mean_with_pressure_raw": mean_raw,
        "mean_with_pressure_fixed": mean_fix
    }).to_csv(out_dir/"mean_curves_hb_band.csv", index=False)

    # ============================
    # RECONSTRUCTION SUCCESS (≥90%) with residual gate
    # ============================
    # def _cosine_sim(a, b):
    #     denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    #     return float(np.dot(a, b) / denom)
    #
    # pressured_idx = np.where(y == 1)[0]
    # success = 0
    # total_pressured = int(len(pressured_idx))
    # band_idx_all = list(range(len(band_wl)))
    #
    # if total_pressured > 0:
    #     for i in pressured_idx:
    #         x_band = X[i].copy()
    #         x_fixed, _ = fix_one(
    #             x_band, band_idx_all,
    #             mu, std, Vk, thr,
    #             template, anchors_idx, edge_n,
    #             blend=device_blend
    #         )
    #         sim = _cosine_sim(x_fixed, template)
    #         # secondary gate: must also pass the training residual threshold
    #         r_fix = residual(x_fixed, mu, std, Vk)
    #         if (sim >= 0.90) and (r_fix < thr):
    #             success += 1
    #     recon_success_rate = success / total_pressured
    # else:
    #     recon_success_rate = np.nan
    #
    # pd.DataFrame({
    #     "metric": ["reconstruction_success_percent"],
    #     "value_percent": [None if np.isnan(recon_success_rate) else recon_success_rate * 100.0],
    #     "n_pressured": [total_pressured],
    #     "n_success": [success]
    # }).to_csv(out_dir / "reconstruction_success.csv", index=False)
    #
    # print(("\nNo pressured samples in validation; reconstruction_success.csv saved with NaN.")
    #       if np.isnan(recon_success_rate)
    #       else f"\nReconstruction success (≥90% sim & residual<thr): {recon_success_rate * 100:.2f}% "
    #            f"({success}/{total_pressured})")

    # ============================
    # MEAN-BAND SIMILARITY (clean vs fixed mean)
    # ============================
    # mean_sim = _cosine_sim(mean_fix, mean_no)
    # pd.DataFrame({
    #     "metric": ["mean_band_cosine_similarity_clean_vs_fixed"],
    #     "value": [mean_sim]
    # }).to_csv(out_dir / "mean_band_similarity.csv", index=False)
    # print(f"Mean Hb-band similarity (clean vs fixed mean): {mean_sim:.3f}")
    #
    # # ============================
    # # FINAL ACCEPTANCE RATE
    # # ============================
    # n_clean = int((y == 0).sum())
    # n_total = int(len(y))
    # n_fixed = 0 if np.isnan(recon_success_rate) else success
    # final_acceptance = (n_clean + n_fixed) / n_total if n_total > 0 else np.nan
    #
    # pd.DataFrame({
    #     "metric": ["final_acceptance_rate_percent"],
    #     "value_percent": [None if np.isnan(final_acceptance) else final_acceptance * 100.0],
    #     "n_total": [n_total],
    #     "n_clean": [n_clean],
    #     "n_fixed": [n_fixed]
    # }).to_csv(out_dir / "final_acceptance.csv", index=False)
    # print(f"Final acceptance rate: {final_acceptance * 100:.2f}% "
    #       f"(clean={n_clean} + fixed={n_fixed} of total={n_total})")

    # ============================
    # Simple similarity helpers
    # ============================
    def _cosine_sim(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def _pearson_r(a, b):
        a = np.asarray(a, float);
        b = np.asarray(b, float)
        a = a - a.mean();
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    # ============================
    # (A) Reconstruction success on pressured samples (≥90% cosine AND residual<thr)
    # ============================
    pressured_idx = np.where(y == 1)[0]
    success = 0
    total_pressured = int(len(pressured_idx))
    band_idx_all = list(range(len(band_wl)))

    if total_pressured > 0:
        for i in pressured_idx:
            x_band = X[i].copy()
            x_fixed, _ = fix_one(
                x_band, band_idx_all,
                mu, std, Vk, thr,
                template, anchors_idx, edge_n,
                blend=device_blend
            )
            sim = _cosine_sim(x_fixed, template)
            r_fix = residual(x_fixed, mu, std, Vk)
            if (sim >= 0.90) and (r_fix < thr):
                success += 1
        recon_success_rate = success / total_pressured
    else:
        recon_success_rate = np.nan

    pd.DataFrame({
        "metric": ["reconstruction_success_percent"],
        "value_percent": [None if np.isnan(recon_success_rate) else recon_success_rate * 100.0],
        "n_pressured": [total_pressured],
        "n_success": [success]
    }).to_csv(out_dir / "reconstruction_success.csv", index=False)

    print(("\nNo pressured samples in validation; reconstruction_success.csv saved with NaN.")
          if np.isnan(recon_success_rate)
          else f"\nReconstruction success (≥90% sim & residual<thr): {recon_success_rate * 100:.2f}% "
               f"({success}/{total_pressured})")

    # ============================
    # (B) Mean-band cosine similarity (fixed mean vs clean mean)
    # ============================
    mean_sim = _cosine_sim(mean_fix, mean_no)
    pd.DataFrame({
        "metric": ["mean_band_cosine_similarity_clean_vs_fixed"],
        "value": [mean_sim]
    }).to_csv(out_dir / "mean_band_similarity.csv", index=False)
    print(f"Mean Hb-band cosine (clean vs fixed mean): {mean_sim:.3f}")

    # ============================
    # (C) Pearson correlation (fixed mean vs clean mean)
    # ============================
    mean_r = _pearson_r(mean_fix, mean_no)
    pd.DataFrame({
        "metric": ["mean_band_pearson_r_clean_vs_fixed"],
        "value": [mean_r]
    }).to_csv(out_dir / "mean_band_correlation.csv", index=False)
    print(f"Mean Hb-band Pearson r (clean vs fixed mean): {mean_r:.3f}")

    # ---- SAVE FIGURES ----
    # Hb-band correction
    plt.figure()
    plt.plot(band_wl, mean_no,  "-o", label="No Pressure",          color="C2")
    plt.plot(band_wl, mean_raw, "-o", label="With Pressure (raw)",  color="C3")
    plt.plot(band_wl, mean_fix, "-o", label="With Pressure (fixed)",color="C0")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Absorbance (a.u.)")
    plt.title("Hb-band correction (Template-Constrained PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"fig_hb_band_correction.png", dpi=220)
    plt.close()

    # Residual histogram
    plt.figure()
    plt.hist([errs[y==0], errs[y==1]], bins=20,
             label=["No Pressure","With Pressure"], alpha=0.7)
    plt.axvline(thr, color="k", linestyle="--", label="Training THR (z-space)")
    plt.xlabel("Reconstruction error (z-space)"); plt.ylabel("Count")
    plt.title("Detection separation (Template-Constrained PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"fig_detection_hist.png", dpi=220)
    plt.close()

    print("Saved CSVs & figures to:", OUT_DIR)

if __name__ == "__main__":
    main()
