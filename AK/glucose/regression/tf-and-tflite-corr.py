import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

# =========================
# 1. Load data
# =========================
dir = "/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose/result/result-ALTY/model-PREV-FLOW/5th_95.5_/wo"
csv_path = f"{dir}/RESULT_10_TF_TFLite_GL_ALTY-data-1st-month_pp_model_2025-05-23_06-53-13.csv"
df = pd.read_csv(csv_path)

y_tf = df["TF_Predicted_GL"]
y_tflite = df["TFLite_Predicted_GL"]

# =========================
# 2. Compute metrics
# =========================
# Pearson correlation
r = np.corrcoef(y_tf, y_tflite)[0, 1]

# R² (how well TFLite matches TF)
r2 = r2_score(y_tf, y_tflite)

print("=== TF vs TFLite Prediction Agreement ===")
print(f"Correlation (r) : {r:.4f}")
print(f"R²             : {r2:.4f}")

# =========================
# 3. Plot TF vs TFLite
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(y_tf, y_tflite, alpha=0.7)

# regression line
m, b = np.polyfit(y_tf, y_tflite, 1)
x_line = np.linspace(y_tf.min(), y_tf.max(), 100)
plt.plot(x_line, m * x_line + b)

plt.xlabel("Based-Model Predicted Glucose")
plt.ylabel("Converted Model Predicted Glucose")
plt.title("3rd model: Based-Model (TensorFlow)\nvs Converted Model (TensorFlow Lite)")

plt.text(
    0.05, 0.95,
    f"R² = {r2:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plot_path = os.path.join(dir, "3rd model: Based-Model (TensorFlow)\nvs Converted Model (TensorFlow Lite).png")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Plot saved to {plot_path}")
