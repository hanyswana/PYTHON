import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import csv
from datetime import datetime

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# File paths
path = '/home/admin-3/PycharmProjects/PythonProject/validation-hb'
data_file = f'{path}/validation-dataset-hb-1k/lablink_1k_hemoglobin_SNV_recove_10.csv'
model_dir = f'{path}/model-recove/corrected-lablink-128-hb_SNV_top_10.parquet_best_model_2024-07-12_14-12-24_R59_88%/'
result_file = f'{path}/result/validation_results_1k.csv'
predictions_file = f'{path}/result/predicted_values_1k.csv'

# Load the data
data = pd.read_csv(data_file)

# Preprocess the data (adjust column names based on your dataset)
target_column = 'Haemoglobin (g/dL)'  # Replace with the actual target column
X = data.drop(columns=['ID', target_column]).astype(np.float32)
y_true = data[target_column].values
ids = data['ID'].values  # Capture IDs for result tracking

# Load the model using TensorFlow's saved_model loader
model = tf.saved_model.load(model_dir)
infer = model.signatures["serving_default"]  # Retrieve the serving signature

# Print model signature for debugging
print("Input Signature:", infer.structured_input_signature)
print("Output Signature:", infer.structured_outputs)

# Prepare input as per signature
input_tensor = {"input_1": tf.convert_to_tensor(X)}  # Replace "input_1" with the actual input key

# Perform inference
predictions = infer(**input_tensor)
y_pred = predictions['regression_head_1'].numpy().flatten()  # Replace 'regression_head_1' with the correct output key
y_pred_rounded = np.round(y_pred, 1)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Categorize residuals
residual = y_true - y_pred
category_1 = np.sum(np.abs(residual) <= 1)
category_2 = np.sum((np.abs(residual) > 1) & (np.abs(residual) <= 2))
category_3 = np.sum(np.abs(residual) > 2)

# Save validation summary to CSV
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
summary_row = {
    "CSV": os.path.basename(data_file),
    "MSE": mse,
    "R2": r2,
    "±1": category_1,
    "±2": category_2,
    ">±2": category_3,
    "Timestamp": timestamp
}

summary_csv_file = result_file
file_exists = os.path.isfile(summary_csv_file)
with open(summary_csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=summary_row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(summary_row)

print(f"Validation summary saved to {summary_csv_file}")

# Save detailed predictions to a separate CSV file
detailed_predictions = pd.DataFrame({
    "ID": ids,
    "Actual Hb": y_true,
    "Predicted Hb": y_pred_rounded
})
detailed_predictions.to_csv(predictions_file, index=False)
print(f"Predicted values saved to {predictions_file}")
