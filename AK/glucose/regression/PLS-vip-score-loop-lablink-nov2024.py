from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
import os

path_dir = '/home/apc-3/PycharmProjects/PythonProjectAK/lablink-nov2024-1k-glucose/dataset-1k-glucose/new/dataset'
base_directory = f'{path_dir}/csv-19/'
pls_directory = f'{path_dir}/'

scores_directory = os.path.join(pls_directory, 'score-PLS')
top10_directory = os.path.join(pls_directory, 'csv-10-PLS')
os.makedirs(scores_directory, exist_ok=True)
os.makedirs(top10_directory, exist_ok=True)

# NEW: collect all VIPs across files
all_vip_rows = []  # NEW

def calculate_vip(pls_model):
    T = pls_model.x_scores_  # Scores
    W = pls_model.x_weights_  # Weights
    Q = pls_model.y_loadings_  # Y loadings
    p, h = W.shape
    vips = np.zeros((p,))

    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i, j] / np.linalg.norm(W[:, j])) * np.sqrt(s[j]) for j in range(h)])
        vips[i] = np.sqrt(p * (np.sum(weight ** 2)) / total_s)
    return vips


for filename in os.listdir(base_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(base_directory, filename)
        data = pd.read_csv(file_path)

        X = data.filter(like='nm')  # Extracting only the spectral data (wavelength columns)
        y = data['glucose']  # Target variable

        pls = PLSRegression(n_components=10)
        pls.fit(X, y)

        # X_scores = pls.transform(X)
        # X_scores_df = pd.DataFrame(X_scores, columns=[f'Component_{i+1}' for i in range(X_scores.shape[1])])
        # output_scores_path = os.path.join(scores_directory, filename.replace('.csv', '_pls.csv'))
        # X_scores_df.to_csv(output_scores_path, index=False)

        vip_scores = calculate_vip(pls)
        vip_scores_dict = dict(zip(X.columns, vip_scores))

        # NEW: add per-wavelength rows with dataset name
        dataset_name = filename.replace('.csv', '')  # NEW
        for wl, vip in vip_scores_dict.items():      # NEW
            all_vip_rows.append({                    # NEW
                'dataset': dataset_name,             # NEW
                'wavelength': wl,                    # NEW
                'vip_score': float(vip)              # NEW
            })                                       # NEW

        top_10_vip_scores = dict(sorted(vip_scores_dict.items(), key=lambda item: item[1], reverse=True)[:10])
        top_10_vip_scores_df = pd.DataFrame(list(top_10_vip_scores.items()), columns=['Wavelength', 'VIP Score'])
        output_vip_path = os.path.join(scores_directory, filename.replace('.csv', '_vip_scores.csv'))
        # top_10_vip_scores_df.to_csv(output_vip_path, index=False)

        first_three_columns = data.columns[:2].tolist()
        columns_to_keep = first_three_columns + list(top_10_vip_scores.keys())
        ordered_columns_to_keep = [col for col in data.columns if col in columns_to_keep]
        top_10_wavelengths_data = data[ordered_columns_to_keep]
        output_top_10_path = os.path.join(top10_directory, filename.replace('.csv', '_top_10.csv'))
        top_10_wavelengths_data.to_csv(output_top_10_path, index=False)

if all_vip_rows:
    all_vip_df = pd.DataFrame(all_vip_rows)
    all_vip_df['rank'] = (
        all_vip_df.sort_values(['dataset', 'vip_score'], ascending=[True, False])
                  .groupby('dataset')
                  .cumcount() + 1
    )
    all_vip_df = all_vip_df.sort_values(['dataset', 'rank'], ascending=[True, True])
    all_vip_df.to_csv(os.path.join(scores_directory, 'all_vip_scores.csv'), index=False)

print("Processing completed for all files.")
