import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os, joblib
from dataclasses import dataclass
from typing import List

# ===================== CONFIG =====================
BASE_DIR   = '/home/apc-3/PycharmProjects/PythonProjectAK/CB-glucose'
MODEL_DIR  = f'{BASE_DIR}/model/1st'                      # where training saved models
MODEL_FILE = 'Lablink_1k_glucose_RAW_CB_best_model.joblib' # <-- pick one model from model/2nd
csv = 'alty_remaining_150'
VAL_CSV = f'{BASE_DIR}/dataset/validation/{csv}.csv'
OUT_DIR    = f'{BASE_DIR}/validation-outputs-ALTY/1st-best-PDS/v1.1-150'         # outputs here
os.makedirs(OUT_DIR, exist_ok=True)

# Calibration Transfer (PDS) model path – trained from CT-PDS v1.1
ct_model_path = f'{BASE_DIR}/model-CT/CT_PDS_Lablink_model_v1.1.joblib'

@dataclass
class PDSModel:
    beta: np.ndarray          # [n_wl, W]
    bias: np.ndarray          # [n_wl]
    win_idx: np.ndarray       # [n_wl, W] int
    wl_names: List[str]       # column names in order
    window: int
    ridge: float
    preprocess_meta: dict     # to record preprocessing order/params

    def transform(self, X_slave: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PDS to a batch DataFrame with the same wavelength columns
        and order as during training.
        """
        X = X_slave[self.wl_names].values.astype(np.float32)
        n, n_wl = X.shape
        W = self.win_idx.shape[1]
        Y = np.zeros_like(X, dtype=np.float32)
        for i in range(n_wl):
            cols = self.win_idx[i]
            Y[:, i] = self.bias[i] + X[:, cols] @ self.beta[i, :W]
        out = pd.DataFrame(Y, columns=self.wl_names, index=X_slave.index)
        return out

    def transform_one(self, row_1d: np.ndarray) -> np.ndarray:
        """
        Apply PDS to a single 1D spectrum in the same wavelength order.
        """
        X = row_1d.astype(np.float32)
        n_wl = X.shape[0]
        W = self.win_idx.shape[1]
        y = np.zeros(n_wl, dtype=np.float32)
        for i in range(n_wl):
            cols = self.win_idx[i]
            y[i] = self.bias[i] + (X[cols] @ self.beta[i, :W])
        return y

# --- Load validation data (must share the same feature schema/order) ---
df = pd.read_csv(VAL_CSV)
X_val  = df.loc[:, '415 nm':'940 nm']
y_true = df.iloc[:, 3]


# === Apply PDS calibration transfer on full 19-band spectra BEFORE wavelength selection ===
pds_model = joblib.load(ct_model_path)
all_spectral_data_ct = pds_model.transform(X_val)

# --- load model and predict ---
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
model = joblib.load(model_path)  # EXACT pipeline used in training
y_pred = np.asarray(model.predict(all_spectral_data_ct)).ravel()

# --- Metrics (same style as training) ---
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
r2   = float(r2_score(y_true, y_pred))
pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])

abs_err = np.abs(y_true - y_pred)
den = np.where(y_true == 0, np.nan, np.abs(y_true))
error_rate_pct = (abs_err / den) * 100.0
accuracy_pct = 100.0 - error_rate_pct
mean_accuracy_pct = float(np.nanmean(accuracy_pct))

# ISO 15197 rule:
#   if actual < 5.55 mmol/L -> |error| <= 0.83 mmol/L
#   else                    -> |error| <= 0.15 * actual
iso_mask = np.where(
    y_true < 5.55,
    abs_err <= 0.83,
    abs_err <= (0.15 * np.abs(y_true))
)
iso_within_pct = float(np.mean(iso_mask) * 100.0)

# --- Save actual vs predicted CSV ---
result_df = pd.DataFrame({
    'actual_glucose': y_true,
    'pred_glucose': y_pred,
    'abs_error': abs_err,
    'iso_within': iso_mask.astype(bool),
    'error_%': error_rate_pct,
    'accuracy_%': accuracy_pct,
})
csv_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(MODEL_FILE))[0] + '_actual_vs_pred.csv')
result_df.to_csv(csv_path, index=False)

# --- Scatter plot (Actual vs Predicted) ---
plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.75, s=40)
mn = float(min(y_true.min(), y_pred.min()))
mx = float(max(y_true.max(), y_pred.max()))
plt.plot([mn, mx], [mn, mx], '--', linewidth=1.5)
plt.title(f'Glucose LABLINK Validation - CB\nr={pearson_r:.3f}  R²={r2:.3f}  RMSE={rmse:.2f}\nISO Within: {iso_within_pct:.1f}%')
plt.xlabel('Actual Glucose (mmol/L)')
plt.ylabel('Predicted Glucose (mmol/L)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(MODEL_FILE))[0] + '_actual_vs_pred_plot.png')
plt.savefig(plot_path, dpi=220)
plt.close()

# --- Save metrics summary CSV ---
metrics_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(MODEL_FILE))[0] + '_metrics.csv')
pd.DataFrame([{
    'Model File': MODEL_FILE,
    'Validation CSV': os.path.basename(MODEL_FILE),
    'RMSE': rmse,
    'R^2': r2,
    'Pearson r': pearson_r,
    'Mean Accuracy %': mean_accuracy_pct,
    'ISO Within %': iso_within_pct
}]).to_csv(metrics_path, index=False)

print('✅ Done.')
print('Outputs:')
print(' - Actual vs Pred CSV :', csv_path)
print(' - Actual vs Pred plot PNG   :', plot_path)
print(' - Metrics CSV        :', metrics_path)