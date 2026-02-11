import subprocess
import os
from pathlib import Path


# Modify selected preprocessing combination
SELECTED_PREPROCESSING = ["None"]  # List : "NormEuc", "NormManh", "SNV", "SavGol", "Baseline", "None"
print(SELECTED_PREPROCESSING)
SAVITZKY_GOLAY_ORDER = 0 # (1 for first derivative, 2 for second derivative, 0 if not used)

# Modify name based on data model and new model name, data 19 wavelength (csv), 10 wavelength (parquet)
# Example of preprocessed model
# model_name= "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline_top_10.parquet_best_model_2024-12-21_01-37-48"
# new_model_name = "model-1"
# data_19_name = "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline"
# data_name = "Lablink_1k_glucose_SNV_Norm_Manh_1st_Deriv_Baseline_top_10"

# # Example of raw model
model_name= "Lablink_1k_glucose_RAW_top_10.parquet_best_model_2025-07-08_14-26-25"
new_model_name = "model-2-v2"
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


# PyInstaller output root (where 4.1–4.5 folders are)
BIN_ROOT_DEFAULT = Path("/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Code/4.Model-converter-bin-pyinstaller/bin")

def get_bin_root() -> Path:
    """
    Portable: if running the compiled master from inside BIN_ROOT,
    auto-detect BIN_ROOT. Otherwise use the default path above.
    """
    here = Path(__file__).resolve()
    # If running compiled master later, __file__ may differ; still safe for .py dev mode
    # We'll also try to detect based on expected folders.
    if (BIN_ROOT_DEFAULT / "4.1.SP-tf-to-tflite").exists():
        return BIN_ROOT_DEFAULT
    return BIN_ROOT_DEFAULT

BIN_ROOT = get_bin_root()


tf_to_tflite_script = f"{BIN_ROOT}/4.1.SP-tf-to-tflite/4.1.SP-tf-to-tflite"
tflite_to_h_script  = f"{BIN_ROOT}/4.2.SP-tflite-to-h/4.2.SP-tflite-to-h"
tfl_ops_script      = f"{BIN_ROOT}/4.3.tfl-operations/4.3.tfl-operations"
wavelength_script   = f"{BIN_ROOT}/4.4.extract-wavelength/4.4.extract-wavelength"
preprocess_script   = f"{BIN_ROOT}/4.5.generate-preprocess/4.5.generate-preprocess"


# def run_script(script_path, *args):
#     """Runs a Python script using subprocess with additional arguments."""
#     try:
#         subprocess.run(["python", script_path, *args], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error running {script_path}: {e}")

def run_script(exe_path, *args):
    """Runs a compiled executable (PyInstaller output) and captures its output."""
    exe_path = str(exe_path)

    try:
        result = subprocess.run(
            [exe_path, *map(str, args)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Output from {exe_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {exe_path}: {e}")
        print(f"Error details:\n{e.stderr}")
        raise


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
