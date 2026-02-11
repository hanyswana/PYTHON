import joblib
import numpy as np
import pandas as pd

# Load training data
file_name = 'SBH-700-DW'
file_path = f'/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR/dataset/ori/parquet/{file_name}.parquet'
df = pd.read_parquet(file_path)
X = df.iloc[:, 3:]  # 19 features
y = df.iloc[:, 2]

# Pick one sample
raw = X.iloc[0].values.reshape(1, -1)

# Load the trained model
file = 'SBH-700-DW_PLSR_best_model'
model_path = f'/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR/model/{file}.joblib'
model = joblib.load(model_path)
scaler = model.named_steps["scaler"]
plsr = model.named_steps["plsr"]

# Apply standardization
x_std = scaler.transform(raw)

# Predict
y_pred = plsr.predict(x_std)
print("Expected prediction:", y_pred[0])

print("Standardized input:", x_std[0])  # ✅ compare this in Arduino

