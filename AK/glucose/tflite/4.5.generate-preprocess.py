import os
import sys

# Preprocessor directory
PREPROCESS_DIR = "/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Preprocess-lib/"

# Mapping of preprocessing methods
ALL_PREPROCESSING_METHODS = {
    "NormEuc": {"header": f"{PREPROCESS_DIR}NormEucLib.h", "source": f"{PREPROCESS_DIR}NormEucLib.cpp"},
    "NormManh": {"header": f"{PREPROCESS_DIR}NormManhLib.h", "source": f"{PREPROCESS_DIR}NormManhLib.cpp"},
    "SNV": {"header": f"{PREPROCESS_DIR}SNV.h", "source": f"{PREPROCESS_DIR}SNV.cpp"},
    "SavGol": {"header": f"{PREPROCESS_DIR}SavGol.h", "source": f"{PREPROCESS_DIR}SavGol.cpp"},
    "Baseline": {"header": f"{PREPROCESS_DIR}BaselineRem.h", "source": f"{PREPROCESS_DIR}BaselineRem.cpp"}
}

# Ensure directory exists
def ensure_directory_exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Remove include statements from header files
def remove_includes_from_file(file_path):
    clean_lines = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                # Skip #include lines
                if not line.strip().startswith("#include"):
                    clean_lines.append(line)
    return clean_lines

# ✅ **Fixes duplicate function declarations & ensures valid C++**
def clean_function_declaration(line):
    """Fix double type declarations (e.g., 'void void')"""
    words = line.split()
    if len(words) > 2 and words[0] == words[1]:  # E.g., "void void function_name"
        return " ".join(words[1:])  # Remove duplicate type
    return line

# ✅ **Generate `preprocess.h`**
def generate_preprocess_header(selected_methods, preprocess_header_path):
    ensure_directory_exists(preprocess_header_path)

    try:
        with open(preprocess_header_path, "w") as f:
            f.write("// Combined Preprocessing Header File\n")
            f.write("#pragma once\n\n")
            f.write("#include <vector>\n")  # Ensure std::vector is available

            if "Baseline" in selected_methods:
                f.write("#include <numeric>\n")  # ✅ Add numeric

            # Store processed function names to avoid duplicates
            function_names = set()

            for method in selected_methods:
                if method in ALL_PREPROCESSING_METHODS:
                    header_file = ALL_PREPROCESSING_METHODS[method]["header"]
                    source_file = ALL_PREPROCESSING_METHODS[method]["source"]

                    # ✅ Write function prototypes
                    f.write(f"\n// -------- {method} Preprocessing (Header) --------\n")
                    clean_header = remove_includes_from_file(header_file)
                    for line in clean_header:
                        line = clean_function_declaration(line)  # Fix duplicate types
                        f.write(line)

                    # ✅ Write function implementations
                    f.write(f"\n// -------- {method} Preprocessing (Source) --------\n")
                    clean_source = remove_includes_from_file(source_file)
                    for line in clean_source:
                        line = clean_function_declaration(line)  # Fix duplicate types
                        f.write(line)

        print(f"✅ Successfully created {preprocess_header_path}")
    except Exception as e:
        print(f"❌ Error writing to {preprocess_header_path}: {e}")

# ✅ **Generate `preprocess_config.h`**
# def generate_preprocess_config(preprocess_config_path, sg_order, selected_methods):
#     ensure_directory_exists(preprocess_config_path)
#
#     try:
#         with open(preprocess_config_path, "w") as f:
#             f.write("// Preprocessing Configuration\n")
#             f.write("#pragma once\n\n")
#
#             # Define preprocessing macros
#             for method in selected_methods:
#                 f.write(f"#define USE_{method.upper()}\n")
#
#             # Define Savitzky-Golay Order
#             if "SavGol" in selected_methods:
#                 f.write(f"#define SAVITZKY_GOLAY_ORDER {sg_order}\n")
#             else:
#                 f.write("#define SAVITZKY_GOLAY_ORDER 0\n")
#
#         print(f"✅ Successfully created {preprocess_config_path}")
#     except Exception as e:
#         print(f"❌ Error writing to {preprocess_config_path}: {e}")

def generate_preprocess_config(preprocess_config_path, sg_order, selected_methods):
    ensure_directory_exists(preprocess_config_path)

    try:
        with open(preprocess_config_path, "w") as f:
            f.write("// Preprocessing Configuration\n")
            f.write("#pragma once\n\n")

            # ✅ Add `#define` macros for preprocessing
            for method in selected_methods:
                f.write(f"#define USE_{method.upper()}\n")

            # ✅ Define preprocessing order as an array
            f.write("\nconst char* PREPROCESS_ORDER[] = {\n")
            for method in selected_methods:
                f.write(f'    "USE_{method.upper()}",\n')
            f.write("    nullptr\n};\n")  # Null-terminated array

            # ✅ Define Savitzky-Golay Order
            if "SavGol" in selected_methods:
                f.write(f"#define SAVITZKY_GOLAY_ORDER {sg_order}\n")
            else:
                f.write("#define SAVITZKY_GOLAY_ORDER 0\n")

        print(f"✅ Successfully created {preprocess_config_path}")
    except Exception as e:
        print(f"❌ Error writing to {preprocess_config_path}: {e}")



# ✅ **Main execution**
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python generate-preprocess.py <preprocess.h path> <preprocess_config.h path> <selected_methods> <sg_order>")
        sys.exit(1)

    preprocess_header_path = sys.argv[1]
    preprocess_config_path = sys.argv[2]
    selected_methods = sys.argv[3].split(",")
    sg_order = int(sys.argv[4])

    generate_preprocess_header(selected_methods, preprocess_header_path)
    generate_preprocess_config(preprocess_config_path, sg_order, selected_methods)
