import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import math, os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# Define the SNV function
def snv(spectra):
    spectra = spectra.values  # Convert DataFrame to NumPy array
    mean_corrected = spectra - np.mean(spectra, axis=1)[:, np.newaxis]
    return mean_corrected / np.std(mean_corrected, axis=1)[:, np.newaxis]

# Define the Normalize Manhattan function
def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Normalize Euclidean function
def apply_norm_euc(X):
    normalizer = Normalizer(norm='l2')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Savitzky-Golay first & second derivative function
def mirror_pad(y, half_window):
    left_pad = y[1:half_window + 1][::-1]
    right_pad = y[-half_window - 1:-1][::-1]
    return np.concatenate((left_pad, y, right_pad))

def SavGol(y, window_size, poly_order, deriv_order=0):
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
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=1),
                        columns=X.columns, index=X.index)

def apply_second_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=2),
                        columns=X.columns, index=X.index)

# Define the Baseline Remover
class BaselineRemover:
    def transform(self, X):
        X = X.to_numpy()  # Ensure it's a NumPy array
        return X - np.mean(X, axis=1, keepdims=True)  # Use keepdims=True for correct broadcasting

def apply_baseline(X):
    baseline_remover = BaselineRemover()
    return pd.DataFrame(baseline_remover.transform(X), columns=X.columns, index=X.index)

# # Define file paths
# csv = 'Hardware_optimization_analysis_v2'
# dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
# data_file = f'{dir}/dataset/10-units-v2/{csv}.csv'

csv = 'alty_remaining_150'
dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
data_file = f'{dir}/dataset/ALTY/{csv}.csv'

# csv = 'Lablink_1k_glucose_RAW'
# dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
# data_file = f'{dir}/dataset/dataset-1k-glucose/ori/{csv}.csv'

# # TensorFlow model paths and output
# mod = 'SNV'
# tf_model = f'Lablink_1k_glucose_{mod}'
# model_dir = 'model-1k-gl-PINTARAI-6th'
# tf_model_dir = f'{dir}/model/model-PINTARAI/{model_dir}/{tf_model}'
# output_file = f'{dir}/result/result-lablink/model-PINTARAI/{model_dir}'
# tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'
#
# # TFLite model paths and output
# tflite_model = f'Lablink_1k_glucose_SNV.tflite'
# tflite_model_dir = f'{dir}/model/model-PINTARAI/{model_dir}/{tflite_model}'

# # TensorFlow model paths and output == 5th model ==
# mod = '2024-12-21_01-37-48'
# tf_model = f'Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline_top_10.parquet_best_model_{mod}'
# model_dir = "4th_95_"
# tf_model_dir = f'{dir}/model/model-PREV-FLOW/{model_dir}/{tf_model}'
# output_file = f'{dir}/result/result-10-units-v2/model-PREV-FLOW/{model_dir}'
# tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'
#
# # TFLite & TFLite model paths and output
# # tflite_model = f'6th_LL_1k_gl_model_{mod}.tflite'
# tflite_model = 'Lablink_1k_glucose_SNV_Manh_1st_D_B_model_2024-12-21_01-37-48.tflite'
# tflite_model_dir = f'{dir}/model/model-PREV-FLOW/{model_dir}/{tflite_model}'

# TensorFlow model paths and output
mod = 'tensorflow'
tf_model = f'model-{mod}'
model_dir = 'Lablink_1k_glucose_SNV_top_10_15'
tf_model_dir = f'{dir}/model/model-PINTARAI-rep/{model_dir}/{tf_model}'
output_file = f'{dir}/result/result-ALTY/model-PINTARAI-rep/{model_dir}'
tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

# TFLite model paths and output
tflite_model = f'Lablink_1k_glucose_SNV_top_10_15.tflite'
tflite_model_dir = f'{dir}/model/model-PINTARAI-rep/{model_dir}/{tflite_model}'

# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']

# Preprocess the data
# norm_euc_data = apply_norm_euc(pd.DataFrame(spectral_data, columns=spectral_data.columns, index=spectral_data.index))
snv_array = snv(spectral_data)
snv_data = pd.DataFrame(snv_array, columns=spectral_data.columns, index=spectral_data.index)
# norm_manh_data = apply_norm_manh(pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index))
# first_derivative_data = apply_first_derivative(norm_manh_data)
# second_derivative_data = apply_second_derivative(snv_data)
# baseline_removed_data = apply_baseline(first_derivative_data)
processed_data = snv_data

# Save the processed 19 spectral data (optional, keep if needed)
processed_19_df = processed_data
processed_19_df.to_csv(tf_output_file_19, index=False)
print(f"Processed 19 spectral data saved to {tf_output_file_19}")

# Select specific spectral bands
# selected_wavelengths = ['415 nm', '445 nm', '480 nm', '610 nm', '630 nm', '645 nm', '680 nm', '860 nm', '900 nm', '940 nm', ] #9th-model-1k-gl-87% (r2_0.61_77_)
# selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '610 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm'] #10th-model-1k-gl-77% (r2_0.61_77_)
# selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '610 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm'] #4th-model-1k-gl-95%
# selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm', '900 nm', '940 nm', ] #5th-model-1k-gl-95.5%
# selected_wavelengths = ['445 nm', '560 nm', '610 nm', '630 nm', '645 nm', '705 nm', '760 nm', '810 nm', '900 nm', '940 nm', ] #4th & 5th-model-PINTARAI
# selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm', '900 nm', '940 nm', ] #6th-model-PINTARAI
selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #snv-model-rep-maxtrials15
selected_data = processed_19_df[selected_wavelengths]


# --- TensorFlow Model Inference ---
tf_model = tf.saved_model.load(tf_model_dir)
tf_infer = tf_model.signatures["serving_default"]
tf_input_tensor = {"input_1": tf.convert_to_tensor(selected_data.astype(np.float32))}
tf_predictions = tf_infer(**tf_input_tensor)
tf_gl_pred = tf_predictions['regression_head_1'].numpy().flatten()


# --- TFLite Model Inference ---
interpreter = tf.lite.Interpreter(model_path=tflite_model_dir)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite Input Details:", input_details)

tflite_gl_preds = []

for i in range(len(selected_data)):
    row_data = selected_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Correctly get and reshape row

    interpreter.set_tensor(input_details[0]['index'], row_data)
    interpreter.invoke()
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
    tflite_gl_pred = tflite_predictions.flatten()[0]
    tflite_gl_preds.append(tflite_gl_pred)

tflite_gl_preds = np.array(tflite_gl_preds)

# --- Save All Predictions into ONE CSV (TF + TFLite + 10 spectral bands) ---
# Keep the first few columns (same as before)
first_two_columns = df.iloc[:, :4]  # adjust if you later want fewer/more

# Combined output file: TF prediction, TFLite prediction, and 10 selected wavelengths
combined_output_file = f'{output_file}/RESULT_10_TF_TFLite_GL_{csv}_pp_model_{mod}.csv'

combined_df = pd.concat(
    [
        first_two_columns.reset_index(drop=True),
        pd.DataFrame({
            'TF_Predicted_GL': np.round(tf_gl_pred, 1),
            'TFLite_Predicted_GL': np.round(tflite_gl_preds, 1),
        }),
        selected_data.reset_index(drop=True)
    ],
    axis=1
)
combined_df.to_csv(combined_output_file, index=False)
print(f"Combined TF + TFLite + 10 spectra saved to {combined_output_file}")

# ===================== METRICS (same as CB-validation) =====================

# Ground-truth glucose – same column index as CB script (4th column)
y_true = df.iloc[:, 3].astype(float).values  # adjust if your GL column moves
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
    f'Glucose LABLINK Validation - AutoKeras TF\n'
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
    f'Glucose LABLINK Validation - AutoKeras TFLite\n'
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
