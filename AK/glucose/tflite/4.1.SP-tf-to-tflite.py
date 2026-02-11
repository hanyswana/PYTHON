import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Read arguments from subprocess call
data_file, model_path, tfl_model_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Load dataset
data = pd.read_parquet(data_file)

# Load TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

# QUANTIZATION (line 16 - 26)
# def representative_dataset_gen():
#     for index, row in data.iterrows():
#         features = row[2:].values.reshape(1, -1).astype(np.float32)
#         # print(features)
#         yield [features]
#
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.float32  # Optional, for quantizing inputs
# converter.inference_output_type = tf.float32  # Optional, for quantizing outputs

# Convert model to TensorFlow Lite
tflite_quant_model = converter.convert()

# Save the converted model
with open(tfl_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"TFLite model saved to: {tfl_model_path}")
