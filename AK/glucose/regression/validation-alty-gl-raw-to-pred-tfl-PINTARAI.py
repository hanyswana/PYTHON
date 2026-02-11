import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os
import math


# Define the SNV function
def snv(spectra):
    spectra = spectra.values  # Convert DataFrame to NumPy array
    mean_corrected = spectra - np.mean(spectra, axis=1)[:, np.newaxis]
    return mean_corrected / np.std(mean_corrected, axis=1)[:, np.newaxis]

# Define the Normalize Manhattan function
def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Euclidean function
def apply_norm_euc(X):
    normalizer = Normalizer(norm='l2')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Savitzky-Golay first derivative function
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


# Define file paths
csv = 'ALTY-data-1st-month'
dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
data_file = f'{dir}/dataset/ALTY/{csv}.csv'

# TensorFlow model paths and output
mod = 'Norm_Manh_SNV_2nd_Deriv_max_trials_15'
tf_model = f'Lablink_1k_glucose_{mod}'
tf_model_dir = f'{dir}/model/model-PINTARAI/model-1k-gl-PINTARAI-4th/{tf_model}'
output_file = f'{dir}/result/result-ALTY/model-PINTARAI/model-1k-gl-PINTARAI-4th/no-pp'
tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

# TFLite & TFLite Quantized model paths and output
tflite_model = f'{tf_model}.tflite'
tflite_model_dir = f'{dir}/model/model-PINTARAI/model-1k-gl-PINTARAI-4th/{tflite_model}'

# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']

# PREPROCESS THE DATA ===== modify according to the trained preprocess =====
norm_manh_data = apply_norm_manh(pd.DataFrame(spectral_data, columns=spectral_data.columns, index=spectral_data.index))
snv_array = snv(norm_manh_data)
snv_data = pd.DataFrame(snv_array, columns=spectral_data.columns, index=spectral_data.index)
# norm_euc_data = apply_norm_euc(pd.DataFrame(spectral_data, columns=spectral_data.columns, index=spectral_data.index))
# first_derivative_data = apply_first_derivative(norm_euc_data)
second_derivative_data = apply_second_derivative(snv_data)
# baseline_removed_data = apply_baseline(first_derivative_data)
processed_data = second_derivative_data

# Save the processed 19 spectral data (optional, keep if needed)
processed_19_df = processed_data
processed_19_df.to_csv(tf_output_file_19, index=False)
print(f"Processed 19 spectral data saved to {tf_output_file_19}")

# Select specific spectral bands
# selected_wavelengths = ['555 nm', '560 nm', '585 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #RAW
selected_wavelengths = ['445 nm', '560 nm', '610 nm', '630 nm', '645 nm', '705 nm', '760 nm', '810 nm', '900 nm', '940 nm'] #model-1k-PINTARAI-3rd & 4th & 5th
# selected_wavelengths = ['445 nm', '560 nm', '610 nm', '630 nm', '645 nm', '705 nm', '760 nm', '810 nm', '900 nm', '940 nm'] #model-1k-PINTARAI-2nd (2025-08-15)
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

# --- Save All Predictions ---
first_two_columns = df.iloc[:, :3]  # Keep the first two columns

# 1. First two columns and all predictions
output_file_1 = f'{output_file}/RESULT_GL_{csv}_pp_ALL_model_{mod}.csv'
predictions_df = pd.concat([first_two_columns], axis=1) # Create a DataFrame with only the first two columns
predictions_df['TF_Predicted_GL'] = np.round(tf_gl_pred, 1)
predictions_df['TFLite_Predicted_GL'] = np.round(tflite_gl_preds, 1)
# predictions_df['Quantized_TFLite_Predicted_GL'] = np.round(quantized_gl_preds, 1)
predictions_df.to_csv(output_file_1, index=False)
print(f"Predictions only saved to {output_file_1}")

# 2. First two columns, TF predictions, and 10 selected wavelengths
output_file_2 = f'{output_file}/RESULT_10_TF_{csv}_pp_model_{mod}.csv'
tf_results_df = pd.concat([first_two_columns, pd.DataFrame({'TF_Predicted_GL': np.round(tf_gl_pred, 1)})], axis=1) # add TF prediction as a DataFrame
tf_results_df = pd.concat([tf_results_df, selected_data], axis=1) # add selected data
tf_results_df.to_csv(output_file_2, index=False)
print(f"TF predictions and 10 spectra saved to {output_file_2}")

# 3. First two columns, TFLite predictions, and 10 selected wavelengths
output_file_3 = f'{output_file}/RESULT_10_TFLite_{csv}_pp_model_{mod}.csv'
tflite_results_df = pd.concat([first_two_columns, pd.DataFrame({'TFLite_Predicted_GL': np.round(tflite_gl_preds, 1)})], axis=1) # add TFLite prediction as a DataFrame
tflite_results_df = pd.concat([tflite_results_df, selected_data], axis=1) # add selected data
tflite_results_df.to_csv(output_file_3, index=False)
print(f"TFLite predictions and 10 spectra saved to {output_file_3}")

# # 4. First two columns, Quantized TFLite predictions, and 10 selected wavelengths
# output_file_4 = f'{output_file}/RESULT_10_TFLite_Quantized_{csv}_pp_model_{mod}.csv'
# quantized_results_df = pd.concat([first_two_columns, pd.DataFrame({'Quantized_TFLite_Predicted_GL': np.round(quantized_gl_preds, 1)})], axis=1) # add Quantized TFLite prediction as a DataFrame
# quantized_results_df = pd.concat([quantized_results_df, selected_data], axis=1) # add selected data
# quantized_results_df.to_csv(output_file_4, index=False)
# print(f"Quantized TFLite predictions and 10 spectra saved to {output_file_4}")