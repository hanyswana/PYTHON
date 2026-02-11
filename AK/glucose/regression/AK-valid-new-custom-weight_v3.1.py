"""
v3.1
Apply CT model v1.1  or v1.1.1 (19 raw input)
Model AK 10 input @ 19 input
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os, math, joblib
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass
from typing import List


def snv(spectra):
    """Standard Normal Variate (SNV) preprocessing with robust NaN handling"""
    spectra = np.asarray(spectra, dtype=np.float32)  # Ensure pure NumPy
    mean_corrected = spectra - np.mean(spectra, axis=1, keepdims=True)
    std_vals = np.std(mean_corrected, axis=1, keepdims=True)

    # Handle zero standard deviation (prevents division by zero -> NaN)
    std_vals = np.where(std_vals == 0, 1, std_vals)  # Replace 0 with 1 to avoid division by zero
    result = mean_corrected / std_vals

    # Additional safety: replace any remaining NaN/inf values
    result = np.where(np.isfinite(result), result, 0)
    return result

# Define the Normalize Manhattan function
def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Euclidean function
def apply_norm_euc(X):
    normalizer = Normalizer(norm='l2')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Savitzky-Golay derivative function
def mirror_pad(y, half_window):
    left_pad = y[1:half_window + 1][::-1]
    right_pad = y[-half_window - 1:-1][::-1]
    return np.concatenate((left_pad, y, right_pad))

def SavGol(y, window_size, poly_order, deriv_order):
    window_size = int(window_size)
    poly_order = int(poly_order)
    if window_size % 2 != 1 or window_size < 1:
        raise ValueError("window_size must be a positive odd number")
    if window_size < poly_order + 2:
        raise ValueError("window_size is too small for the polynomial order")
    half_window = (window_size - 1) // 2
    b = np.array([[k ** i for i in range(poly_order + 1)] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b)[deriv_order] * (math.factorial(deriv_order))
    y_padded = mirror_pad(y, half_window)
    return np.convolve(m[::-1], y_padded, mode='valid')


def apply_first_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=9, poly_order=2, deriv_order=1),
                        columns=X.columns, index=X.index)

def apply_second_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=9, poly_order=2, deriv_order=2),
                        columns=X.columns, index=X.index)

# Define the Baseline Remover
class BaselineRemover:
    def transform(self, X):
        X = X.to_numpy()  # Ensure it's a NumPy array
        return X - np.mean(X, axis=1, keepdims=True)  # Use keepdims=True for correct broadcasting

def apply_baseline(X):
    baseline_remover = BaselineRemover()
    return pd.DataFrame(baseline_remover.transform(X), columns=X.columns, index=X.index)

# Define file paths ===== ALTY =====
csv = 'alty_remaining_150'
dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
data_file = f'{dir}/dataset/ALTY/{csv}.csv'

# Calibration Transfer (PDS) model path – trained from CT-PDS v1.1
ct_model_path = f'{dir}/model-CT/PDS/model-v3/raw-v1.1.1/CT_PDS_Lablink_model_v1.1.1.joblib'

# TensorFlow model paths and output == 1st model ==
# mod = 'R2_0.94_97_'
# tf_model = f'Lablink_1k_glucose_RAW_Norm_Manh_SNV_1st_Deriv_batch32_{mod}'
# model_dir = "1st_R2_0.94_97_"
# tf_model_dir = f'{dir}/model/model-v3/{model_dir}/{tf_model}'
# output_file = f'{dir}/result/result-ALTY/model-v3/{model_dir}/OC/v1.1-150'
# tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

# TensorFlow model paths and output == 2nd model ==
mod = 'R20.5_89_'
tf_model = f'Lablink_1k_glucose_RAW_SNV_batch4_{mod}'
model_dir = "2nd_R20.5_89_"
tf_model_dir = f'{dir}/model/model-v3/{model_dir}/{tf_model}'
output_file = f'{dir}/result/result-ALTY/model-v3/{model_dir}/PDS/raw-v1.1.1-150'
tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# TFLite & TFLite model paths and output
tflite_model = f'{model_dir}.tflite'
tflite_model_dir = f'{dir}/model/model-v3/{model_dir}/{tflite_model}'

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

# @dataclass
# class OrthCTModel:
#     mu_slave: np.ndarray        # [n_wl]
#     mu_master: np.ndarray       # [n_wl]
#     P_dev: np.ndarray           # [n_wl, k]
#     wl_names: List[str]
#     n_components: int
#     preprocess_meta: dict
#
#     def transform(self, X_slave: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply orthogonal correction to a batch of slave spectra (DataFrame).
#         """
#         X = X_slave[self.wl_names].values.astype(np.float64)
#         Xc = X - self.mu_slave[None, :]           # center by slave mean
#
#         # Projection onto device-effect subspace (z: [n_samples, k])
#         Z = Xc @ self.P_dev                        # (n,p) @ (p,k) = (n,k)
#
#         # Reconstruct device-effect in spectral space
#         X_dev = Z @ self.P_dev.T                   # (n,k) @ (k,p) = (n,p)
#
#         # Remove device effect and shift to master mean
#         X_corr = (Xc - X_dev) + self.mu_master[None, :]
#
#         out = pd.DataFrame(X_corr.astype(np.float32),
#                            columns=self.wl_names,
#                            index=X_slave.index)
#         return out
#
#     def transform_one(self, row_1d: np.ndarray) -> np.ndarray:
#         """
#         Apply orthogonal correction to a single slave spectrum (1D ndarray).
#         """
#         x = row_1d.astype(np.float64)
#         xc = x - self.mu_slave
#
#         z = xc @ self.P_dev                 # [k]
#         x_dev = self.P_dev @ z              # [p]  (since P_dev: [p,k])
#
#         x_corr = (xc - x_dev) + self.mu_master
#         return x_corr.astype(np.float32)

# Load the dataset
df = pd.read_csv(data_file)
all_spectral_data = df.loc[:, '415 nm':'940 nm']

# === Apply PDS calibration transfer on full 19-band spectra BEFORE wavelength selection ===
pds_model = joblib.load(ct_model_path)
all_spectral_data_ct = pds_model.transform(all_spectral_data)

# Preprocess the data
# norm_manh_data = apply_norm_manh(pd.DataFrame(all_spectral_data_ct, columns=all_spectral_data_ct.columns, index=all_spectral_data_ct.index))
# norm_euc_data = apply_norm_euc(pd.DataFrame(spectral_data, columns=spectral_data.columns, index=spectral_data.index))
snv_array = snv(all_spectral_data_ct)
snv_data = pd.DataFrame(snv_array, columns=all_spectral_data_ct.columns, index=all_spectral_data_ct.index)
# first_derivative_data = apply_first_derivative(snv_data)
# second_derivative_data = apply_second_derivative(snv_data)
# baseline_removed_data = apply_baseline(first_derivative_data)
processed_data_ori = snv_data

# === SELECT THE 10 TRAINING BANDS FIRST ===
# selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '705 nm', '760 nm','810 nm', '860 nm', '900 nm']  # 1st_R2_0.94_97_
selected_wavelengths = ['555 nm', '560 nm', '585 nm', '630 nm', '645 nm', '705 nm', '810 nm', '860 nm', '900 nm', '940 nm']  # 2nd_R20.5_89_
spectral_data = processed_data_ori[selected_wavelengths].copy()
processed_data = spectral_data

# == TensorFlow Model Inference (CUSTOM WEIGHT) ==
def clinical_weighted_mse(y_true, y_pred):
    # Weights: Hypoglycemic(4.0), Normal(1.0), Prediabetic(3.0), Diabetic(5.0)
    weights = tf.where(
        y_true < 4.1, 4.0,
        tf.where(y_true <= 8.1, 1.0,
                 tf.where(y_true <= 10.0, 3.0, 5.0))
    )
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def clinical_weighted_mae(y_true, y_pred):
    weights = tf.where(
        y_true < 4.1, 4.0,
        tf.where(y_true <= 8.1, 1.0,
                 tf.where(y_true <= 10.0, 3.0, 5.0))
    )
    return tf.reduce_mean(weights * tf.abs(y_true - y_pred))

# Register for any Keras deserialization that occurs at load time
get_custom_objects().update({
    'clinical_weighted_mse': clinical_weighted_mse,
    'clinical_weighted_mae': clinical_weighted_mae
})

def load_model_with_customs(model_path: str):
    """Load a Keras model (SavedModel dir or .h5/.keras) with custom losses registered."""
    with tf.keras.utils.custom_object_scope({
        'clinical_weighted_mse': clinical_weighted_mse,
        'clinical_weighted_mae': clinical_weighted_mae
    }):
        return tf.keras.models.load_model(model_path)

# --- TensorFlow Model Inference (Keras load with custom losses) ---
# Try loading as a Keras model first (ensures identical graph & any custom objs)
try:
    keras_model = load_model_with_customs(tf_model_dir)
    tf_gl_pred = keras_model.predict(
        processed_data.astype(np.float32),
        verbose=0
    ).reshape(-1)
except Exception as e:
    print("Keras load failed, falling back to tf.saved_model.load. Reason:", str(e))
    # Fallback: use the SavedModel signature (works when model exported with TF’s SavedModel)
    tf_model = tf.saved_model.load(tf_model_dir)
    tf_infer = tf_model.signatures.get("serving_default")
    if tf_infer is None:
        raise RuntimeError("No 'serving_default' signature found in SavedModel.")
    tf_input_tensor = {"input_1": tf.convert_to_tensor(processed_data.astype(np.float32))}
    tf_predictions = tf_infer(**tf_input_tensor)

    # NOTE: update the key below if your output node name differs
    # (you can print(list(tf_predictions.keys())) to confirm)
    output_key = list(tf_predictions.keys())[0]
    tf_gl_pred = tf_predictions[output_key].numpy().reshape(-1)


# --- TFLite Model Inference ---
interpreter = tf.lite.Interpreter(model_path=tflite_model_dir)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite Input Details:", input_details)

tflite_gl_preds = []

for i in range(len(processed_data)):
    row_data = processed_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Correctly get and reshape row

    interpreter.set_tensor(input_details[0]['index'], row_data)
    interpreter.invoke()
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
    pred_scalar = float(np.squeeze(tflite_predictions))   # ensure scalar
    tflite_gl_preds.append(pred_scalar)

tflite_gl_pred = np.array(tflite_gl_preds, dtype=np.float32)

# --- Save All Predictions ---
first_two_columns = df.iloc[:, :4]  # Keep the first four columns == ALTY ==
# first_two_columns = df.iloc[:, :2]  # Keep the first two columns == LABLINK ==

# Combined output file: TF prediction, TFLite prediction, and 10 selected wavelengths
combined_output_file = f'{output_file}/RESULT_10_TF_TFLite_GL_{csv}_pp_model_{mod}.csv'

combined_df = pd.concat(
    [
        first_two_columns.reset_index(drop=True),
        pd.DataFrame({
            'TF_Predicted_GL': np.round(tf_gl_pred, 1),
            # 'TFLite_Predicted_GL': np.round(tflite_gl_pred, 1),
        }),
        processed_data.reset_index(drop=True)
    ],
    axis=1
)
combined_df.to_csv(combined_output_file, index=False)
print(f"Combined TF + TFLite + 10 spectra saved to {combined_output_file}")


# --- Save internal preprocessed spectral data (TF/Keras Penultimate Layer) ---
# === SIMPLE: Save 10-D internal features from TFLite (after model) ===
def _pick_10d_tensor(interp, input_index):
    """Pick the first non-input tensor with shape (1,10) or (-1,10)."""
    for t in interp.get_tensor_details():
        shp = list(t.get('shape_signature', [])) or list(t.get('shape', []))
        if t['index'] != input_index and (shp == [1, 10] or shp == [-1, 10]):
            return t['index']
    return None

def save_after_model_features(interp, X_df, first_cols_df, out_csv_path):
    """Feeds X_df into the model and saves the tapped 10-D internal tensor to CSV."""
    inp = interp.get_input_details()[0]
    input_idx = inp['index']
    tap_idx = _pick_10d_tensor(interp, input_idx)

    if tap_idx is None:
        print("⚠️ No 10-D internal tensor found. Nothing saved.")
        return False

    feats = []
    for i in range(len(X_df)):
        row = X_df.iloc[i:i+1].astype(np.float32).values
        interp.set_tensor(input_idx, row)
        interp.invoke()
        feats.append(np.squeeze(interp.get_tensor(tap_idx)))

    feats = np.vstack(feats)
    feat_cols = [f"after_model_{j+1}" for j in range(feats.shape[1])]
    out_df = pd.concat([first_cols_df.reset_index(drop=True),
                        pd.DataFrame(feats, columns=feat_cols)], axis=1)
    out_df.to_csv(out_csv_path, index=False)
    print(f"✅ Saved after-model features → {out_csv_path}")
    return True

# Save internal 10-D features (after model) alongside the first columns
internal_tfl_csv = f"{output_file}/TFLite_after_model_features_{csv}_{mod}.csv"
_ = save_after_model_features(interpreter, processed_data, first_two_columns, internal_tfl_csv)


# ===================== METRICS =====================

# Ground-truth glucose – same column index as CB script (4th column)
y_true = df.iloc[:, 3].astype(float).values  # == ALTY ==
# y_true = df.iloc[:, 1].astype(float).values  # == LABLINK ==
print(y_true)

def compute_metrics(y_true, y_pred):
    """Compute RMSE, R², Pearson r, per-sample error, accuracy and ISO-15197 %."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Per-sample error
    abs_err = np.abs(y_true - y_pred)
    den = np.where(y_true == 0, np.nan, np.abs(y_true))
    error_rate_pct = (abs_err / den) * 100.0
    accuracy_pct = 100.0 - error_rate_pct
    mean_accuracy_pct = float(np.nanmean(accuracy_pct))

    # ISO 15197 condition:
    # If actual < 5.55 mmol/L  -> |error| <= 0.83 mmol/L
    # Else                     -> |error| <= 0.15 * actual
    iso_mask = np.where(
        y_true < 5.55,
        abs_err <= 0.83,
        abs_err <= (0.15 * np.abs(y_true))
    )
    iso_within_pct = float(np.mean(iso_mask) * 100.0)

    return {
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "abs_err": abs_err,
        "error_%": error_rate_pct,
        "accuracy_%": accuracy_pct,
        "mean_accuracy_%": mean_accuracy_pct,
        "iso_mask": iso_mask,
        "iso_within_%": iso_within_pct,
    }

# Make sure output directory exists
os.makedirs(output_file, exist_ok=True)

# ---------- Metrics for TF (SavedModel) ----------
metrics_tf = compute_metrics(y_true, tf_gl_pred)

tf_result_df = pd.DataFrame({
    'actual_glucose': y_true,
    'pred_glucose': tf_gl_pred,
    'abs_error': metrics_tf['abs_err'],
    'iso_within': metrics_tf['iso_mask'].astype(bool),
    'error_%': metrics_tf['error_%'],
    'accuracy_%': metrics_tf['accuracy_%'],
})

tf_csv_path = os.path.join(output_file, f'AK_TF_{csv}_actual_vs_pred.csv')
tf_result_df.to_csv(tf_csv_path, index=False)

# Scatter plot (Actual vs Predicted) for TF
plt.figure(figsize=(7, 6))
plt.scatter(y_true, tf_gl_pred, alpha=0.75, s=40)
mn = float(min(y_true.min(), tf_gl_pred.min()))
mx = float(max(y_true.max(), tf_gl_pred.max()))
plt.plot([mn, mx], [mn, mx], '--', linewidth=1.5)
plt.title(
    f'Glucose ALTY Validation - AutoKeras TF\n'
    f'r={metrics_tf["pearson_r"]:.3f}  R²={metrics_tf["r2"]:.3f}  '
    f'RMSE={metrics_tf["rmse"]:.2f}\nISO Within: {metrics_tf["iso_within_%"]:.1f}%'
)
plt.xlabel('Actual Glucose (mmol/L)')
plt.ylabel('Predicted Glucose (mmol/L)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
tf_plot_path = os.path.join(output_file, f'AK_TF_{csv}_actual_vs_pred_plot.png')
plt.savefig(tf_plot_path, dpi=220)
plt.close()

# Metrics summary CSV for TF
tf_metrics_path = os.path.join(output_file, f'AK_TF_{csv}_metrics.csv')
pd.DataFrame([{
    'Model Type': 'AutoKeras TF',
    'TF Model Dir': os.path.basename(tf_model_dir),
    'Validation CSV': os.path.basename(data_file),
    'RMSE': metrics_tf['rmse'],
    'R^2': metrics_tf['r2'],
    'Pearson r': metrics_tf['pearson_r'],
    'Mean Accuracy %': metrics_tf['mean_accuracy_%'],
    'ISO Within %': metrics_tf['iso_within_%'],
}]).to_csv(tf_metrics_path, index=False)


# ---------- Metrics for TFLite ----------
metrics_tfl = compute_metrics(y_true, tflite_gl_preds)

tfl_result_df = pd.DataFrame({
    'actual_glucose': y_true,
    'pred_glucose': tflite_gl_preds,
    'abs_error': metrics_tfl['abs_err'],
    'iso_within': metrics_tfl['iso_mask'].astype(bool),
    'error_%': metrics_tfl['error_%'],
    'accuracy_%': metrics_tfl['accuracy_%'],
})

tfl_csv_path = os.path.join(output_file, f'AK_TFLite_{csv}_actual_vs_pred.csv')
tfl_result_df.to_csv(tfl_csv_path, index=False)

# Scatter plot (Actual vs Predicted) for TFLite
plt.figure(figsize=(7, 6))
plt.scatter(y_true, tflite_gl_preds, alpha=0.75, s=40)
mn = float(min(y_true.min(), tflite_gl_preds.min()))
mx = float(max(y_true.max(), tflite_gl_preds.max()))
plt.plot([mn, mx], [mn, mx], '--', linewidth=1.5)
plt.title(
    f'Glucose ALTY Validation - AutoKeras TFLite\n'
    f'r={metrics_tfl["pearson_r"]:.3f}  R²={metrics_tfl["r2"]:.3f}  '
    f'RMSE={metrics_tfl["rmse"]:.2f}\nISO Within: {metrics_tfl["iso_within_%"]:.1f}%'
)
plt.xlabel('Actual Glucose (mmol/L)')
plt.ylabel('Predicted Glucose (mmol/L)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
tfl_plot_path = os.path.join(output_file, f'AK_TFLite_{csv}_actual_vs_pred_plot.png')
plt.savefig(tfl_plot_path, dpi=220)
plt.close()

# Metrics summary CSV for TFLite
tfl_metrics_path = os.path.join(output_file, f'AK_TFLite_{csv}_metrics.csv')
pd.DataFrame([{
    'Model Type': 'AutoKeras TFLite',
    'TFLite Model File': os.path.basename(tflite_model_dir),
    'Validation CSV': os.path.basename(data_file),
    'RMSE': metrics_tfl['rmse'],
    'R^2': metrics_tfl['r2'],
    'Pearson r': metrics_tfl['pearson_r'],
    'Mean Accuracy %': metrics_tfl['mean_accuracy_%'],
    'ISO Within %': metrics_tfl['iso_within_%'],
}]).to_csv(tfl_metrics_path, index=False)

print("✅ Metrics for AutoKeras TF & TFLite saved:")
print(" - TF  actual vs pred CSV :", tf_csv_path)
print(" - TF  plot PNG           :", tf_plot_path)
print(" - TF  metrics CSV        :", tf_metrics_path)
print(" - TFL actual vs pred CSV :", tfl_csv_path)
print(" - TFL plot PNG           :", tfl_plot_path)
print(" - TFL metrics CSV        :", tfl_metrics_path)
