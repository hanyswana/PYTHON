import pandas as pd
import numpy as np
import tensorflow as ker
from sklearn.preprocessing import Normalizer
import os, math, json
# from tensorflow.keras.models import load_model, model_from_json
from keras.models import load_model, model_from_json
import autokeras as ak


# Define the SNV function
def snv(spectra):
    spectra = spectra.values  # Convert DataFrame to NumPy array
    mean_corrected = spectra - np.mean(spectra, axis=1)[:, np.newaxis]
    return mean_corrected / np.std(mean_corrected, axis=1)[:, np.newaxis]

# Define the Normalize Manhattan function
def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)

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

# Define file paths
csv = 'ALTY-data-1st-month'
dir = '/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose'
data_file = f'{dir}/dataset/ALTY/{csv}.csv'

# TensorFlow model paths and output
mod = 'best_model_90%_keras'
ker_model = f'Lablink_1k_glucose_RAW_top_10_SNV_Norm_Manh_2nd_Deriv_Baseline/{mod}'
ker_model_dir = f'{dir}/model-1k-gl-90%-keras/{ker_model}'
output_file = f'{dir}/result/result-ALTY/model-1k-gl-90%-keras'
ker_output_file_19 = f'{output_file}/RESULT_19_keras_{csv}_pp_model_{mod}.csv'

# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']

# Select specific spectral bands
# selected_wavelengths = ['445 nm', '560 nm', '610 nm', '630 nm', '645 nm', '705 nm', '760 nm', '810 nm', '900 nm', '940 nm'] #model-1k-gl-90%-keras (pp)
selected_wavelengths = ['555 nm', '560 nm', '585 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #model-1k-gl-90%-keras (raw)
raw_selected_data = spectral_data[selected_wavelengths]

# Preprocess the data
# snv_data = snv(spectral_data)
snv_array = snv(raw_selected_data)
snv_data = pd.DataFrame(snv_array, columns=raw_selected_data.columns, index=raw_selected_data.index)
# norm_manh_data = apply_norm_manh(pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index))
norm_manh_data = apply_norm_manh(pd.DataFrame(snv_data))
# norm_euc_data = apply_norm_euc(pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index))
# first_derivative_data = apply_first_derivative(norm_manh_data)
second_derivative_data = apply_second_derivative(norm_manh_data)
baseline_removed_data = apply_baseline(second_derivative_data)
processed_data = baseline_removed_data

# Save the processed 19 spectral data (optional, keep if needed)
processed_19_df = processed_data
processed_19_df.to_csv(ker_output_file_19, index=False)
print(f"Processed 19 spectral data saved to {ker_output_file_19}")

# --- Keras H5 Model Inference ---
# Load model architecture from config.json
with open(f"{ker_model_dir}/config.json", "r") as json_file:
    model_json = json_file.read()

keras_model = model_from_json(model_json)
keras_model.load_weights(f"{ker_model_dir}/model.weights.h5")
keras_gl_pred = keras_model.predict(processed_data.astype(np.float32)).flatten()

# # Select specific spectral bands
# # selected_wavelengths = ['445 nm', '560 nm', '610 nm', '630 nm', '645 nm', '705 nm', '760 nm', '810 nm', '900 nm', '940 nm'] #model-1k-gl-90%-keras (pp)
# selected_wavelengths = ['555 nm', '560 nm', '585 nm', '630 nm', '645 nm', '680 nm', '810 nm', '860 nm', '900 nm', '940 nm'] #model-1k-gl-90%-keras (raw)
# selected_data = processed_19_df[selected_wavelengths]
#
# # --- Keras H5 Model Inference ---
# # Load model architecture from config.json
# with open(f"{ker_model_dir}/config.json", "r") as json_file:
#     model_json = json_file.read()
#
# keras_model = model_from_json(model_json)
# keras_model.load_weights(f"{ker_model_dir}/model.weights.h5")
# keras_gl_pred = keras_model.predict(selected_data.astype(np.float32)).flatten()

# --- Save All Predictions ---
first_two_columns = df.iloc[:, :3]  # Keep the first two columns

# 1. First two columns and all predictions
output_file_1 = f'{output_file}/RESULT_GL_{csv}_pp_ALL_model_{mod}.csv'
predictions_df = pd.concat([first_two_columns], axis=1) # Create a DataFrame with only the first two columns
predictions_df['Keras_Predicted_GL'] = np.round(keras_gl_pred, 1)
predictions_df.to_csv(output_file_1, index=False)
print(f"Predictions only saved to {output_file_1}")

# 2. First two columns, ker predictions, and 10 selected wavelengths
output_file_2 = f'{output_file}/RESULT_10_ker_{csv}_pp_model_{mod}.csv'
ker_results_df = pd.concat([first_two_columns, pd.DataFrame({'ker_Predicted_GL': np.round(keras_gl_pred, 1)})], axis=1) # add ker prediction as a DataFrame
ker_results_df = pd.concat([ker_results_df, processed_data], axis=1) # add selected data
ker_results_df.to_csv(output_file_2, index=False)
print(f"Keras predictions and 10 spectra saved to {output_file_2}")

