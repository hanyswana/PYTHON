import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os
import math
from tensorflow.keras.utils import get_custom_objects


# == Preprocess ==
# Define the SNV function
def snv(spectra):
    spectra = spectra.values  # Convert DataFrame to NumPy array
    mean_corrected = spectra - np.mean(spectra, axis=1)[:, np.newaxis]
    return mean_corrected / np.std(mean_corrected, axis=1)[:, np.newaxis]

# Define the Normalize Manhattan function
def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Euclidean Manhattan function
def apply_norm_euc(X):
    normalizer = Normalizer(norm='l2')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

# Define the Savitzky-Golay first derivative function
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

def apply_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=1),
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
dir = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/NEW-validation/validation-1k-glucose'
data_file = f'{dir}/dataset/ALTY/{csv}.csv'

# TensorFlow model paths and output
mod = 'r2_0.61_77_'
tf_model = f'Lablink_1k_glucose_RAW_top_10_Norm_Euc_SNV_1st_Deriv_Baseline_batch64_{mod}'
model_dir = "10th_r2_0.61_77_"
tf_model_dir = f'{dir}/model-NEW-FLOW/{model_dir}/{tf_model}'
output_file = f'{dir}/result/result-ALTY/model-NEW-FLOW/{model_dir}fixed_pp'
tf_output_file_19 = f'{output_file}/RESULT_19_tf_{csv}_pp_model_{mod}.csv'

# TFLite & TFLite model paths and output
# tflite_model = f'6th_LL_1k_gl_model_{mod}.tflite'
tflite_model = '10th_Lablink_1k_glucose_RAW_top_10_Norm_Euc_SNV_1st_Deriv_Baseline_batch64_r2_0.61_77_.tflite'
tflite_model_dir = f'{dir}/model-NEW-FLOW/{model_dir}/{tflite_model}'

# TFLite & TFLite Quantized model paths and output
# quantized_tflite_model = f'lablink_1k_hb_Euc_SNV_1st_D_model_2025-01-22_16-32-31_Q.tflite' # Replace with your quantized model name
# quantized_tflite_model_dir = f'/home/admin-3/PycharmProjects/PythonProjectAK-REVA/validation-1k-hb/model-1k-hb/{quantized_tflite_model}'


# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']

# Preprocess the data
norm_euc_data = apply_norm_euc(pd.DataFrame(spectral_data, columns=spectral_data.columns, index=spectral_data.index))
snv_array = snv(norm_euc_data)
snv_data = pd.DataFrame(snv_array, columns=spectral_data.columns, index=spectral_data.index)
# norm_manh_data = apply_norm_manh(pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index))
derivative_data = apply_derivative(snv_data)
baseline_removed_data = apply_baseline(derivative_data)
processed_data = baseline_removed_data

# Save the processed 19 spectral data (optional, keep if needed)
processed_19_df = processed_data
processed_19_df.to_csv(tf_output_file_19, index=False)
print(f"Processed 19 spectral data saved to {tf_output_file_19}")

# Select specific spectral bands
selected_wavelengths = ['555 nm', '560 nm', '585 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #RAW
# selected_wavelengths = ['415 nm', '445 nm', '480 nm', '610 nm', '630 nm', '645 nm', '680 nm', '860 nm', '900 nm', '940 nm', ] #9th-model-1k-gl-787% (r2_0.61_77_)
# selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '610 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm'] #10th-model-1k-gl-77% (r2_0.61_77_)
# selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #model-1k-gl-85% (2025-10-14_12-33-53)
# selected_wavelengths = ['560 nm', '590 nm', '630 nm', '645 nm', '680 nm', '730 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #model-1k-gl-85% (batch16_r2_0.52_acc_85_)
# selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '610 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm'] #model-1k-95 (2024-12-21_01-37-48)
selected_data = processed_19_df[selected_wavelengths]


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
        selected_data.astype(np.float32),
        verbose=0
    ).reshape(-1)
except Exception as e:
    print("Keras load failed, falling back to tf.saved_model.load. Reason:", str(e))
    # Fallback: use the SavedModel signature (works when model exported with TF’s SavedModel)
    tf_model = tf.saved_model.load(tf_model_dir)
    tf_infer = tf_model.signatures.get("serving_default")
    if tf_infer is None:
        raise RuntimeError("No 'serving_default' signature found in SavedModel.")
    tf_input_tensor = {"input_1": tf.convert_to_tensor(selected_data.astype(np.float32))}
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

for i in range(len(selected_data)):
    row_data = selected_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Correctly get and reshape row

    interpreter.set_tensor(input_details[0]['index'], row_data)
    interpreter.invoke()
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])
    tflite_gl_pred = tflite_predictions.flatten()[0]
    tflite_gl_preds.append(tflite_gl_pred)

tflite_gl_preds = np.array(tflite_gl_preds)


# # --- Quantized TFLite Model Inference ---
# quantized_interpreter = tf.lite.Interpreter(model_path=quantized_tflite_model_dir)
# quantized_interpreter.allocate_tensors()
# quantized_input_details = quantized_interpreter.get_input_details()
# quantized_output_details = quantized_interpreter.get_output_details()
# print("Quantized TFLite Input Details:", quantized_input_details)

# quantized_gl_preds = []
# for i in range(len(selected_data)):
#     row_data = selected_data.iloc[i].values.reshape(1, -1).astype(np.float32) # Input data must be float32 even for quantized model
#     quantized_interpreter.set_tensor(quantized_input_details[0]['index'], row_data)
#     quantized_interpreter.invoke()
#     quantized_predictions = quantized_interpreter.get_tensor(quantized_output_details[0]['index'])
#     quantized_gl_pred = quantized_predictions.flatten()[0]
#     quantized_gl_preds.append(quantized_gl_pred)
# quantized_gl_preds = np.array(quantized_gl_preds)


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