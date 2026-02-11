import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def main():
    base_dir = '/home/apc-3/PycharmProjects/PythonProjectAK/PLSR-glucose'
    data_dir = f'{base_dir}/dataset/dataset-1k-glucose/ori/parquet/0-1'
    model_out_dir = f'{base_dir}/model/2nd'
    score_out_csv = f'{base_dir}/score/2nd/glucose_PLSR_scores.csv'  # single combined score file
    plots_dir = f'{base_dir}/score-report/2nd'

    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(score_out_csv), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Loop through every parquet in ori/
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.parquet'):
            continue

        file_name = os.path.splitext(fname)[0]
        file_path = os.path.join(data_dir, fname)
        print(f'\n=== Processing: {file_name} ===')

        # --- your original data selection (kept as-is) ---
        def read_parquet_robust(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                print(f"Warning: {path} could not be read as Parquet ({e}). Skipped.")
                return None

        data = read_parquet_robust(file_path)
        if data is None:
            continue

        X = data.iloc[:, 2:]
        y = data.iloc[:, 1]
        # print('X:', X)
        # print('y:', y)


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('plsr', PLSRegression(scale=False))
        ])

        parameters = {
            # 'plsr__n_components': range(2, 10),
            'plsr__n_components': range(2, min(20, X_train.shape[1])),
            'plsr__max_iter': [500, 1000, 2000],
            'plsr__tol': [1e-05, 1e-06, 1e-04]
        }

        grid_search = GridSearchCV(
            pipeline, parameters, cv=5, scoring='neg_mean_squared_error',
            n_jobs=1, pre_dispatch='1*n_jobs', verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # --- Pearson correlation (y_test vs y_pred) ---
        # Flatten to 1D just in case y_pred is shape (n, 1)
        y_true_1d = np.asarray(y_test).ravel()
        y_pred_1d = np.asarray(y_pred).ravel()
        pearson_r = np.corrcoef(y_true_1d, y_pred_1d)[0, 1]

        # --- ISO 15197 error/accuracy ---
        abs_err = np.abs(y_true_1d - y_pred_1d)
        den = np.where(y_true_1d == 0, np.nan, np.abs(y_true_1d))
        error_rate_pct = (abs_err / den) * 100.0
        accuracy_pct = 100.0 - error_rate_pct
        mean_accuracy_pct = float(np.nanmean(accuracy_pct))

        # ISO rule:
        #   if actual < 5.55 mmol/L  -> |error| <= 0.83 mmol/L
        #   else                      -> |error| <= 0.15 * actual
        iso_mask = np.where(
            y_true_1d < 5.55,
            abs_err <= 0.83,
            abs_err <= (0.15 * np.abs(y_true_1d))
        )

        iso_within_pct = float(np.mean(iso_mask) * 100.0)

        # --- Save scatter plot for this dataset ---
        plot_path = os.path.join(plots_dir, f'{file_name}_corr.png')
        plt.figure(figsize=(6.4, 4.8))
        plt.scatter(y_true_1d, y_pred_1d, alpha=0.7, s=30)
        min_v = float(min(y_true_1d.min(), y_pred_1d.min()))
        max_v = float(max(y_true_1d.max(), y_pred_1d.max()))
        plt.plot([min_v, max_v], [min_v, max_v], '--')  # ideal y=x line
        plt.title(f'Correlation: r={pearson_r:.3f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        # Save model per dataset (same naming pattern)
        model_file = f'{file_name}_PLSR_best_model.joblib'
        model_filename = os.path.join(model_out_dir, model_file)
        joblib.dump(best_model, model_filename)

        # Append one row into the single combined score file
        new_row = pd.DataFrame([{
            'Model Filename': model_file,
            'Dataset': file_name,
            'PLSR n_components': grid_search.best_params_['plsr__n_components'],
            'Max Iter': grid_search.best_params_['plsr__max_iter'],
            'Tolerance': grid_search.best_params_['plsr__tol'],
            'RMSE': rmse,
            'R^2': r2,
            'Pearson r': pearson_r,
            'Mean Accuracy %': mean_accuracy_pct,
            'ISO Within %': iso_within_pct,
        }])

        # --- Overwrite or update score file ---
        if os.path.exists(score_out_csv):
            existing = pd.read_csv(score_out_csv)
            # remove any previous entry with same Dataset or model
            existing = existing[existing['Dataset'] != file_name]
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined.to_csv(score_out_csv, index=False)
            print("✓ Score CSV updated / overwritten entry")
        else:
            new_row.to_csv(score_out_csv, mode='w', index=False)
            print("✓ Created new score CSV")


if __name__ == "__main__":
    main()
