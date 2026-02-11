import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os

# Define the Manhattan normalization
def normalize_manhattan(data):
    normalizer = Normalizer(norm='l1')
    return normalizer.fit_transform(data)

# Define spectral selection
def select_spectral_bands(data, selected_wavelengths):
    return data[selected_wavelengths]

# File paths
csv = 'Hardware_Optimization_Analysis'
data_file = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/dataset/10-units-v2/{csv}.csv'
mod = '2025-01-22_16-32-31'
model = f'lablink_1k_hemoglobin_Norm_Euc_SNV_1st_Deriv_top_10.parquet_best_model_{mod}'
model_dir = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/model-1k-hb/{model}'
output_file_19 = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/result/10-units-v2/RESULT_19_{csv}_manh_model_{mod}.csv'
output_file_10 = f'/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-hb/result/10-units-v2/RESULT_10_{csv}_manh_model_{mod}.csv'


# Load the dataset
df = pd.read_csv(data_file)
spectral_data = df.loc[:, '415 nm':'940 nm']  # Select columns from 415 nm to 940 nm
actual_hemoglobin = df['Unit']  # Replace with the actual column name for hemoglobin values

# Normalize the data using Manhattan normalization
normalized_data = normalize_manhattan(spectral_data.values)
normalized_19_df = pd.DataFrame(normalized_data, columns=spectral_data.columns, index=df.index)

# Save the normalized 19 spectral data
normalized_19_df.to_csv(output_file_19, index=False)
print(f"Normalized 19 spectral data saved to {output_file_19}")

# Select specific spectral bands
selected_wavelengths = ['515 nm', '560 nm', '610 nm', '630 nm', '645 nm', '680 nm', '705 nm', '860 nm', '900 nm', '940 nm']
selected_data = select_spectral_bands(normalized_19_df, selected_wavelengths)

# Load the TensorFlow model
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]

# Prepare input for the model
input_tensor = {"input_1": tf.convert_to_tensor(selected_data.astype(np.float32))}

# Perform inference
predictions = infer(**input_tensor)
hemoglobin_pred = predictions['regression_head_1'].numpy().flatten()  # Replace output key as needed

# Save the selected 10 spectral bands with actual and predicted hemoglobin
output_10_df = selected_data.copy()
output_10_df.insert(0, 'Predicted_Hemoglobin', hemoglobin_pred)
output_10_df.insert(0, 'Actual_Hemoglobin', actual_hemoglobin)
output_10_df.to_csv(output_file_10, index=False)
print(f"Selected 10 spectral bands and predictions saved to {output_file_10}")
