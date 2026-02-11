import subprocess
import os


# Modify selected preprocessing combination
SELECTED_PREPROCESSING = ["NormEuc"]  # List : "NormEuc", "NormManh", "SNV", "SavGol", "Baseline", "None"
print(SELECTED_PREPROCESSING)
SAVITZKY_GOLAY_ORDER = 0 # (1 for first derivative, 2 for second derivative, 0 if not used)

# Modify name based on data model and new model name, data 19 wavelength (csv), 10 wavelength (parquet)
# Example of preprocessed model
# model_name= "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline_top_10.parquet_best_model_2024-12-21_01-37-48"
# new_model_name = "model-1"
# data_19_name = "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline"
# data_name = "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline_top_10"

# # Example of raw model
model_name= "Lablink_1k_glucose_RAW_top_10_Norm_Euc_batch16_r2_0.33_87_"
new_model_name = "Q_9th_Lablink_1k_gl_RAW_top_10_Norm_Euc_r2_0.33_87_"
data_19_name = "Lablink_1k_glucose_RAW"
data_name = "Lablink_1k_glucose_RAW_top_10"


dir = "/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module"
path = f"Model-glucose/{new_model_name}"
data_file = f"{dir}/Data-glucose/pls/dataset-1k-pls-top10-parquet/{data_name}.parquet"
wavelength_19_file = f"{dir}/Data-glucose/ori/{data_19_name}.csv"  # Full 19 wavelengths

os.makedirs(os.path.dirname(path), exist_ok=True)

model_path = f"{dir}/{path}/model-tensorflow/{model_name}"
tfl_model_path = f"{dir}/{path}/model-converted-and-output/{new_model_name}.tflite"
header_model_path = f"{dir}/{path}/model-converted-and-output/{new_model_name}.h"
ops_header_output_path = f"{dir}/{path}/model-converted-and-output/OPS_{new_model_name}.h"
wavelength_10_header = f"{dir}/{path}/model-converted-and-output/WL_{new_model_name}.h"
preprocess_header_path = f"{dir}/{path}/model-converted-and-output/PP_{new_model_name}.h"
preprocess_config_path = f"{dir}/{path}/model-converted-and-output/PP_config_{new_model_name}.h"  # New config file for SG derivative

tf_to_tflite_script = f"{dir}/Code/4.Model-converter/4.1.SP-tf-to-tflite.py"
tflite_to_h_script = f"{dir}/Code/4.Model-converter/4.2.SP-tflite-to-h.py"
tfl_ops_script = f"{dir}/Code/4.Model-converter/4.3.tfl-operations.py"
wavelength_script = f"{dir}/Code/4.Model-converter/4.4.extract-wavelength.py"  # New wavelength extraction script
preprocess_script = f"{dir}/Code/4.Model-converter/4.5.generate-preprocess.py"  # New script


# def run_script(script_path, *args):
#     """Runs a Python script using subprocess with additional arguments."""
#     try:
#         subprocess.run(["python", script_path, *args], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error running {script_path}: {e}")

def run_script(script_path, *args):
    """Runs a Python script using subprocess and captures its output."""
    try:
        result = subprocess.run(
            ["python", script_path, *args],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Output from {script_path}:\n{result.stdout}")  # ✅ Show script output
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(f"Error details:\n{e.stderr}")  # ✅ Show error details


if __name__ == "__main__":
    print("Converting TensorFlow model to TensorFlow Lite...")
    run_script(tf_to_tflite_script, data_file, model_path, tfl_model_path)

    print("Converting TensorFlow Lite model to TensorFlow Lite Micro (.h)...")
    run_script(tflite_to_h_script, tfl_model_path, header_model_path)

    print("Extracting operations from TensorFlow Lite model...")
    run_script(tfl_ops_script, tfl_model_path, ops_header_output_path)  # Extract ops

    print("Extracting Wavelength Indices...")
    run_script(wavelength_script, wavelength_19_file, data_file, wavelength_10_header)

    print("Generating preprocessing header and configuration...")
    run_script(preprocess_script, preprocess_header_path, preprocess_config_path, ",".join(SELECTED_PREPROCESSING), str(SAVITZKY_GOLAY_ORDER))

    print("Conversion process completed.")
