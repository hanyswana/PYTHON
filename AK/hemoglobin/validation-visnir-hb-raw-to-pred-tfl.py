import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os
import math


# Define the SNV function
def snv(spectra):
    spectra = spectra.values  # Convert the pandas DataFrame to a numpy array
    mean_corrected = spectra - np.mean(spectra, axis=1)[:, np.newaxis]  # Use numpy's slicing and reshape
    return mean_corrected / np.std(mean_corrected, axis=1)[:, np.newaxis]  # Ensure std is reshaped for each row

# Define the Normalize Euclidean function
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
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=1),
                        columns=X.columns, index=X.index)


# Define file paths
csv = 'Minitab-test-spectral'
data_file = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/dataset/10-units-REVA-dataset-hb/{csv}.csv'

# TensorFlow model paths and output
mod = '2025-01-22_16-32-31'
tf_model = f'lablink_1k_hemoglobin_Norm_Euc_SNV_1st_Deriv_top_10.parquet_best_model_{mod}'
tf_model_dir = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/model-1k-hb/{tf_model}'
output_file = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/result/10-units-result-hb-corrected'
tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

# TFLite & TFLite Quantized model paths and output
tflite_model = f'lablink_1k_hb_Euc_SNV_1st_D_model_2025-01-22_16-32-31.tflite'
tflite_model_dir = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/model-1k-hb/{tflite_model}'

quantized_tflite_model = f'lablink_1k_hb_Euc_SNV_1st_D_model_2025-01-22_16-32-31_Q.tflite' # Replace with your quantized model name
quantized_tflite_model_dir = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/model-1k-hb/{quantized_tflite_model}'


# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']

# # Preprocess the data
# snv_data = snv(spectral_data)
# snv_data_df = pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index)
# normalized_data = apply_norm_euc(snv_data_df)
# processed_data = apply_first_derivative(snv_data_df)

# Preprocess the data
normalized_data = apply_norm_euc(spectral_data)
snv_data = snv(normalized_data)
snv_data_df = pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index)
processed_data = apply_first_derivative(snv_data_df)

# Save the processed 19 spectral data (optional, keep if needed)
processed_19_df = processed_data
processed_19_df.to_csv(tf_output_file_19, index=False)
print(f"Processed 19 spectral data saved to {tf_output_file_19}")

# Select specific spectral bands
selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '630 nm', '645 nm', '730 nm', '810 nm', '860 nm', '900 nm']
selected_data = processed_19_df[selected_wavelengths]


# --- TensorFlow Model Inference ---
tf_model = tf.saved_model.load(tf_model_dir)
tf_infer = tf_model.signatures["serving_default"]
tf_input_tensor = {"input_1": tf.convert_to_tensor(selected_data.astype(np.float32))}
tf_predictions = tf_infer(**tf_input_tensor)
tf_hemoglobin_pred = tf_predictions['regression_head_1'].numpy().flatten()


# --- TFLite Model Inference ---
interpreter = tf.lite.Interpreter(model_path=tflite_model_dir)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite Input Details:", input_details)

tflite_hemoglobin_preds = []

for i in range(len(selected_data)):
    row_data = selected_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Correctly get and reshape row

    interpreter.set_tensor(input_details[0]['index'], row_data)
    interpreter.invoke()
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
    tflite_hemoglobin_pred = tflite_predictions.flatten()[0]
    tflite_hemoglobin_preds.append(tflite_hemoglobin_pred)

tflite_hemoglobin_preds = np.array(tflite_hemoglobin_preds)


# --- Quantized TFLite Model Inference ---
quantized_interpreter = tf.lite.Interpreter(model_path=quantized_tflite_model_dir)
quantized_interpreter.allocate_tensors()
quantized_input_details = quantized_interpreter.get_input_details()
quantized_output_details = quantized_interpreter.get_output_details()
print("Quantized TFLite Input Details:", quantized_input_details)

quantized_hemoglobin_preds = []
for i in range(len(selected_data)):
    row_data = selected_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Input data must be float32 even for quantized model
    quantized_interpreter.set_tensor(quantized_input_details[0]['index'], row_data)
    quantized_interpreter.invoke()
    quantized_predictions = quantized_interpreter.get_tensor(quantized_output_details[0]['index'])
    quantized_hemoglobin_pred = quantized_predictions.flatten()[0]
    quantized_hemoglobin_preds.append(quantized_hemoglobin_pred)
quantized_hemoglobin_preds = np.array(quantized_hemoglobin_preds)


# --- Save All Predictions ---
first_two_columns = df.iloc[:, :2]  # Keep the first two columns

# 1. First two columns and all predictions
output_file_1 = f'{output_file}/RESULT_Hb_{csv}_pp_ALL_model_{mod}.csv'
predictions_df = pd.concat([first_two_columns], axis=1) # Create a DataFrame with only the first two columns
predictions_df['TF_Predicted_Hemoglobin'] = np.round(tf_hemoglobin_pred, 1)
predictions_df['TFLite_Predicted_Hemoglobin'] = np.round(tflite_hemoglobin_preds, 1)
predictions_df['Quantized_TFLite_Predicted_Hemoglobin'] = np.round(quantized_hemoglobin_preds, 1)
predictions_df.to_csv(output_file_1, index=False)
print(f"Predictions only saved to {output_file_1}")

# 2. First two columns, TF predictions, and 10 selected wavelengths
output_file_2 = f'{output_file}/RESULT_10_TF_{csv}_pp_model_{mod}.csv'
tf_results_df = pd.concat([first_two_columns, pd.DataFrame({'TF_Predicted_Hemoglobin': np.round(tf_hemoglobin_pred, 1)})], axis=1) # add TF prediction as a DataFrame
tf_results_df = pd.concat([tf_results_df, selected_data], axis=1) # add selected data
tf_results_df.to_csv(output_file_2, index=False)
print(f"TF predictions and 10 spectra saved to {output_file_2}")

# 3. First two columns, TFLite predictions, and 10 selected wavelengths
output_file_3 = f'{output_file}/RESULT_10_TFLite_{csv}_pp_model_{mod}.csv'
tflite_results_df = pd.concat([first_two_columns, pd.DataFrame({'TFLite_Predicted_Hemoglobin': np.round(tflite_hemoglobin_preds, 1)})], axis=1) # add TFLite prediction as a DataFrame
tflite_results_df = pd.concat([tflite_results_df, selected_data], axis=1) # add selected data
tflite_results_df.to_csv(output_file_3, index=False)
print(f"TFLite predictions and 10 spectra saved to {output_file_3}")

# 4. First two columns, Quantized TFLite predictions, and 10 selected wavelengths
output_file_4 = f'{output_file}/RESULT_10_TFLite_Quantized_{csv}_pp_model_{mod}.csv'
quantized_results_df = pd.concat([first_two_columns, pd.DataFrame({'Quantized_TFLite_Predicted_Hemoglobin': np.round(quantized_hemoglobin_preds, 1)})], axis=1) # add Quantized TFLite prediction as a DataFrame
quantized_results_df = pd.concat([quantized_results_df, selected_data], axis=1) # add selected data
quantized_results_df.to_csv(output_file_4, index=False)
print(f"Quantized TFLite predictions and 10 spectra saved to {output_file_4}")