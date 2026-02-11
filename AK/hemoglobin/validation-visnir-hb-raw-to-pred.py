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
data_file = f'/home/admin-3/PycharmProjects/PythonProjectAK-REVA/validation-1k-hb/dataset/10-units-REVA-hb/{csv}.csv'
mod = '2025-01-22_16-32-31'
model = f'lablink_1k_hemoglobin_Norm_Euc_SNV_1st_Deriv_top_10.parquet_best_model_{mod}'
model_dir = f'/home/admin-3/PycharmProjects/PythonProjectAK-REVA/validation-1k-hb/model-1k-hb/{model}'
output_file_19 = f'/home/admin-3/PycharmProjects/PythonProjectAK-REVA/validation-1k-hb/result/10-units/RESULT_19_{csv}_preprocessed_model_{mod}.csv'
output_file_10 = f'/home/admin-3/PycharmProjects/PythonProjectAK-REVA/validation-1k-hb/result/10-units/RESULT_10_{csv}_preprocessed_model_{mod}.csv'

# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']  # Select columns from 415 nm to 940 nm

# Preprocess the data: Apply SNV, Normalize Euclidean, and First Derivative
snv_data = snv(spectral_data)
# Convert back to DataFrame for further processing
snv_data_df = pd.DataFrame(snv_data, columns=spectral_data.columns, index=spectral_data.index)
normalized_data = apply_norm_euc(snv_data_df)
processed_data = apply_first_derivative(normalized_data)

# Save the processed 19 spectral data
processed_19_df = processed_data
processed_19_df.to_csv(output_file_19, index=False)
print(f"Processed 19 spectral data saved to {output_file_19}")

# Select specific spectral bands
selected_wavelengths = ['445 nm', '560 nm', '585 nm', '590 nm', '630 nm', '645 nm', '730 nm', '810 nm', '860 nm', '900 nm']
selected_data = processed_19_df[selected_wavelengths]

# Load the TensorFlow model
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]

# Prepare input for the model
input_tensor = {"input_1": tf.convert_to_tensor(selected_data.astype(np.float32))}

# Perform inference
predictions = infer(**input_tensor)
hemoglobin_pred = predictions['regression_head_1'].numpy().flatten()  # Replace output key as needed

# Save the selected 10 spectral bands with actual and predicted hemoglobin
# output_10_df = selected_data.copy()
# output_10_df.insert(0, 'Predicted_Hemoglobin', hemoglobin_pred)
# output_10_df.to_csv(output_file_10, index=False)

# Save the selected 10 spectral bands with actual and predicted hemoglobin
first_two_columns = df.iloc[:, :2]  # Select first two columns
output_10_df = pd.concat([first_two_columns, selected_data], axis=1)
output_10_df.insert(2, 'Predicted_Hemoglobin', np.round(hemoglobin_pred, 1))
output_10_df.to_csv(output_file_10, index=False)
print(f"Selected 10 spectral bands and predictions saved to {output_file_10}")
