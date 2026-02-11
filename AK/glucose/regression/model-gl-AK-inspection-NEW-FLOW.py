import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------ USER SET THIS: SavedModel folder path ------------
model_dir = "/home/apc-3/PycharmProjects/PythonProjectAK/validation-1k-glucose/"
model_path = f"{model_dir}/model-NEW-FLOW/10th_r2_0.61_77_/Lablink_1k_glucose_RAW_top_10_Norm_Euc_SNV_1st_Deriv_Baseline_batch64_r2_0.61_77_"
# ---------------------------------------------------------------

train_data_path = f"{model_dir}/dataset/dataset-1k-glucose/pls/dataset-1k-glucose-pls-top10-parquet/Lablink_1k_glucose_RAW_top_10.parquet"
# -------------------------------------

print("\n✅ Loading SavedModel...")
model = tf.saved_model.load(model_path)
fn = model.signatures["serving_default"]

# === INPUT/OUTPUT ===
in_sig = fn.structured_input_signature[1]
print("\n=== INPUT SPEC ===")
for name, spec in in_sig.items():
    print(f" {name}: shape={list(spec.shape)}, dtype={spec.dtype}")

print("\n=== OUTPUT SPEC ===")
for name, tensor in fn.structured_outputs.items():
    print(f" {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

# === TRAINING DATA — only 10 nm columns ===
df = pd.read_parquet(train_data_path)
X_raw = df[[c for c in df.columns if "nm" in c]].astype(np.float32).values
print("\n✅ Loaded training features:", X_raw.shape)

# === GRAPH OPS — check preprocessing existence ===
ops = [op.type for op in fn.graph.get_operations()]
counts = Counter(ops)

print("\n=== TOP OPERATIONS ===")
for op,n in counts.most_common(15):
    print(f"{op:25s} x{n}")

preprocess_ops = ["L2Normalize","Mean","Sub","Rsqrt","Conv1D","MatMul"]
inside_preprocess = any(op in counts for op in preprocess_ops)

if inside_preprocess:
    print("\n✅ Model preprocessing detected inside graph.")
else:
    print("\n⚠️ No preprocessing operations found inside model.")

# === TEST RAW vs PREPROCESSED INFERENCE ===
print("\n🔍 Running inference test...")

# RAW input
raw_pred = fn(input_1=tf.constant(X_raw))["regression_head_1"].numpy().squeeze()

# SIMULATED preprocess (not exact!!) — just to test sensitivity
# Z-score (SNV-like)
scaler = StandardScaler()
X_snv = scaler.fit_transform(X_raw)
pp_pred = fn(input_1=tf.constant(X_snv.astype(np.float32)))["regression_head_1"].numpy().squeeze()

corr = np.corrcoef(raw_pred, pp_pred)[0,1]
print(f"\n📌 Correlation RAW vs SNV prediction = {corr:.3f}")

if corr < 0.90:
    print("❌ Output changes significantly → preprocessing REQUIRED before input to model.")
else:
    print("✅ Output stable → preprocessing likely included inside model.")

print("\n🎯 Inspection Completed.\n")
