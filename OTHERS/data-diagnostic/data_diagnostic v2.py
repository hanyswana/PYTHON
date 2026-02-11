import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, pearsonr

# === SETUP ===
DIR = '/home/apc-3/PycharmProjects/PythonProjectAK/DATA-diagnostic'
BASE_DIR_LABLINK = f'{DIR}/dataset-lablink-alty/all-24/lablink'
BASE_DIR_ALTY = f'{DIR}/dataset-lablink-alty/all-24/alty'
OUT_DIR  = os.path.join(DIR, 'output/all-24')
os.makedirs(OUT_DIR, exist_ok=True)

def load_xy(path):
    df = pd.read_csv(path)
    wl_cols = [c for c in df.columns if 'nm' in c]
    if 'glucose' in df.columns:
        y = df['glucose'].values
    else:
        y = df.iloc[:, 1].values
    X = df[wl_cols].copy()
    return X, y

# === LOOP THROUGH ALL MATCHED PAIRS ===
results = []

# --- Make sure subfolders exist ---
mean_dir = os.path.join(OUT_DIR, 'mean-plot')
pca_dir  = os.path.join(OUT_DIR, 'pca-plot')
os.makedirs(mean_dir, exist_ok=True)
os.makedirs(pca_dir,  exist_ok=True)

## === MATCHING SECTION ===

def extract_tag_lablink(filename):
    # everything after 'Lablink_1k_glucose_'
    base = os.path.splitext(filename)[0]
    return base.split('Lablink_1k_glucose_', 1)[1]

def extract_tag_alty(filename):
    # everything after 'ALTY-data-1st-month_'
    base = os.path.splitext(filename)[0]
    return base.split('ALTY-data-1st-month_', 1)[1]

lablink_files = [f for f in os.listdir(BASE_DIR_LABLINK) if f.startswith('Lablink_1k_glucose_') and f.endswith('.csv')]
alty_files    = [f for f in os.listdir(BASE_DIR_ALTY) if f.startswith('ALTY-data-1st-month_') and f.endswith('.csv')]

lab_tags  = {extract_tag_lablink(f): f for f in lablink_files}
alty_tags = {extract_tag_alty(f):    f for f in alty_files}

common_tags = sorted(set(lab_tags.keys()) & set(alty_tags.keys()))
print(f"✅ Found {len(common_tags)} matched Lablink–ALTY dataset pairs")

if not common_tags:
    print("⚠ No matches found — check prefix names or ensure both datasets share the same suffix.")


results = []

for tag in common_tags:
    lab_f = lab_tags[tag]
    alt_f = alty_tags[tag]

    # Load both datasets
    Xtr, ytr = load_xy(os.path.join(BASE_DIR_LABLINK, lab_f))
    Xva, yva = load_xy(os.path.join(BASE_DIR_ALTY, alt_f))
    common_cols = [c for c in Xtr.columns if c in Xva.columns]
    if not common_cols:
        print(f"⚠ No common wavelength columns for tag {tag}. Skipping.")
        continue
    Xtr, Xva = Xtr[common_cols], Xva[common_cols]

    # == MEAN ± STD ==
    m_tr, s_tr = Xtr.mean(0).values, Xtr.std(0).values
    m_va, s_va = Xva.mean(0).values, Xva.std(0).values
    wls = np.array([float(c.split()[0]) for c in common_cols])

    plt.figure(figsize=(8,5))
    plt.plot(wls, m_tr, label='Lablink mean')
    plt.fill_between(wls, m_tr - s_tr, m_tr + s_tr, alpha=0.2)
    plt.plot(wls, m_va, label='ALTY mean')
    plt.fill_between(wls, m_va - s_va, m_va + s_va, alpha=0.2)
    plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity'); plt.legend()
    plt.title(f'{tag}: Spectral Means ±1 SD')
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, f'{tag}_mean_std.png'), dpi=200)
    plt.close()

    # == PCA ==
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Z = scaler.fit_transform(np.vstack([Xtr.values, Xva.values]))
    pca = PCA(n_components=2)
    PC = pca.fit_transform(Z)
    evr = pca.explained_variance_ratio_
    evr_pc1, evr_pc2 = evr[0]*100, evr[1]*100
    evr_total = evr_pc1 + evr_pc2
    ntr = len(Xtr)

    plt.figure(figsize=(6,5))
    plt.scatter(PC[:ntr,0], PC[:ntr,1], s=15, alpha=0.6, label='Lablink')
    plt.scatter(PC[ntr:,0], PC[ntr:,1], s=15, alpha=0.6, label='ALTY')
    plt.title(f'{tag}: PCA (PC1={evr_pc1:.1f}%, PC2={evr_pc2:.1f}%, Total={evr_total:.1f}%)')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pca_dir, f'{tag}_pca.png'), dpi=200)
    plt.close()

    # == Similarity metrics ==
    from scipy.spatial.distance import cosine
    from scipy.stats import ks_2samp, pearsonr
    cos_sim = 1 - cosine(m_tr, m_va)
    per_wl_corr = pearsonr(m_tr, m_va)[0]
    ss_res = np.sum((m_tr - m_va) ** 2)
    ss_tot = np.sum((m_tr - np.mean(m_tr)) ** 2)
    r2_mean_spectra = 1 - (ss_res / ss_tot)
    ks_stats = [ks_2samp(Xtr[c], Xva[c]).pvalue for c in common_cols]
    ks_fraction_diff = float(np.mean(np.array(ks_stats) < 0.05))

    results.append({
        'preprocess_tag': tag,
        'cosine_similarity': float(cos_sim),
        'pearson_r': float(per_wl_corr),
        'r2_mean_spectra': float(r2_mean_spectra),
        'ks_fraction_p<0.05': ks_fraction_diff,
        'pca_total_%': float(evr_total),
    })

# Save CSV (even if empty, so the printed path is real)
df_results = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, 'all_datasets_diagnostics_summary.csv')
df_results.to_csv(csv_path, index=False)
print(f"✅ Saved all results to {csv_path}  (rows: {len(df_results)})")
