import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, pearsonr

DIR = '/home/apc-3/PycharmProjects/PythonProjectAK/DATA-diagnostic'
TRAIN_CSV = f'{DIR}/dataset-lablink-alty/raw/Lablink_1k_glucose_RAW.csv'
VAL_CSV   = f'{DIR}/dataset-lablink-alty/raw/ALTY-data-1st-month.csv'
OUT_DIR   = f'{DIR}/output'
os.makedirs(OUT_DIR, exist_ok=True)

def load_xy(path):
    df = pd.read_csv(path)
    wl_cols = [c for c in df.columns if 'nm' in c]  # wavelength columns
    if 'glucose' in df.columns:
        y = df['glucose'].values
    else:
        y = df.iloc[:, 1].values
    X = df[wl_cols].copy()
    return X, y

Xtr, ytr = load_xy(TRAIN_CSV)
Xva, yva = load_xy(VAL_CSV)

# Align columns
common = [c for c in Xtr.columns if c in Xva.columns]
Xtr = Xtr[common]
Xva = Xva[common]

# 1) Mean ± std overlay
m_tr, s_tr = Xtr.mean(0).values, Xtr.std(0).values
m_va, s_va = Xva.mean(0).values, Xva.std(0).values
wls = np.array([float(c.split()[0]) for c in common])

plt.figure(figsize=(8,5))
plt.plot(wls, m_tr, label='Train mean')
plt.fill_between(wls, m_tr - s_tr, m_tr + s_tr, alpha=0.2)
plt.plot(wls, m_va, label='Val mean')
plt.fill_between(wls, m_va - s_va, m_va + s_va, alpha=0.2)
plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity'); plt.legend()
plt.title('Spectral Means ±1 SD (Train vs Val)')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'mean_std_overlay.png'), dpi=220); plt.close()

# 2) PCA overlap
scaler = StandardScaler()
X_all = np.vstack([Xtr.values, Xva.values])
Z = scaler.fit_transform(X_all)
pca = PCA(n_components=2).fit(Z)
PC = pca.transform(Z)
evr = pca.explained_variance_ratio_  # [PC1, PC2]
evr_pc1, evr_pc2 = evr[0]*100, evr[1]*100
evr_total = (evr_pc1 + evr_pc2)
ntr = len(Xtr)

plt.figure(figsize=(6,5))
plt.scatter(PC[:ntr,0], PC[:ntr,1], s=15, alpha=0.6, label='Train')
plt.scatter(PC[ntr:,0], PC[ntr:,1], s=15, alpha=0.6, label='Val')
plt.title(f'PCA: Train vs Val (PC1={evr_pc1:.1f}%, PC2={evr_pc2:.1f}%, Total={evr_total:.1f}%)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pca_overlap.png'), dpi=220)
plt.close()

# 3) Similarity metrics
cos_sim = 1 - cosine(m_tr, m_va)
per_wl_corr = pearsonr(m_tr, m_va)[0]

# --- Compute R² (coefficient of determination) between mean spectra ---
ss_res = np.sum((m_tr - m_va) ** 2)
ss_tot = np.sum((m_tr - np.mean(m_tr)) ** 2)
r2_mean_spectra = 1 - (ss_res / ss_tot)


# KS test across wavelengths (distributional diff)
ks_stats = []
for c in common:
    ks_stats.append(ks_2samp(Xtr[c], Xva[c]).pvalue)
ks_fraction_diff = float(np.mean(np.array(ks_stats) < 0.05))

# 4) Label distribution summary
summary = {
    'n_train': len(ytr), 'n_val': len(yva),
    'train_glucose_min': float(np.min(ytr)), 'train_glucose_max': float(np.max(ytr)),
    'val_glucose_min': float(np.min(yva)),   'val_glucose_max': float(np.max(yva)),
    'cosine_similarity_mean_spectra': float(cos_sim),
    'pearson_r_mean_spectra': float(per_wl_corr),
    'r2_mean_spectra': float(r2_mean_spectra),
    'fraction_wavelengths_KS_p<0.05': ks_fraction_diff,
    'pca_total_pc1_pc2_%': float(evr_total)
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'diagnostics_summary.csv'), index=False)
print(summary)
