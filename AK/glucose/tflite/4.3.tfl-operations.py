import sys
import tensorflow as tf


def format_op_name(op_name):
    """Formats the operation name to match the required Arduino function format."""
    words = op_name.lower().split("_")  # Convert to lowercase and split by underscores
    formatted_name = "Add" + "".join(word.capitalize() for word in words)  # Capitalize each word and prefix with 'Add'
    return formatted_name


def extract_tflite_operations(tflite_model_path, header_output_path):
    """Extracts operations from a TensorFlow Lite model and saves them in a header file for Arduino."""

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    ops_list = set()

    for op in interpreter._get_ops_details():
        op_name = op["op_name"]

        # Ignore "DELEGATE" as it's not a real operation
        if op_name == "DELEGATE":
            continue

        formatted_op = format_op_name(op_name)
        ops_list.add(formatted_op)

    # Check for fused activation functions (ReLU, etc.)
    for tensor_details in interpreter.get_tensor_details():
        if "activation" in tensor_details and tensor_details["activation"] == "RELU":
            ops_list.add("AddRelu")  # Explicitly add ReLU if found

    # Debugging: Print tensor details to check for fused activations
    # print("=== Tensor Details ===")
    # for tensor_details in interpreter.get_tensor_details():
    #     print(tensor_details)
    # print("======================")


    num_ops = len(ops_list)  # Total number of operations

    # Generate the C++ header content
    header_content = f"""#ifndef TFLITE_OPERATIONS_H
#define TFLITE_OPERATIONS_H

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#define NUM_TFLITE_OPS {num_ops}

inline void RegisterOps(tflite::MicroMutableOpResolver<NUM_TFLITE_OPS> &resolver) {{
"""

    for op in sorted(ops_list):
        header_content += f"    resolver.{op}();\n"

    header_content += "}\n\n#endif // TFLITE_OPERATIONS_H\n"

    # Write to header file
    with open(header_output_path, 'w') as file:
        file.write(header_content)

    print(f"Operations extracted and saved to: {header_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_tflite_ops_to_h.py <tflite_model_path> <header_output_path>")
        sys.exit(1)

    tflite_model_path = sys.argv[1]
    header_output_path = sys.argv[2]
    extract_tflite_operations(tflite_model_path, header_output_path)
