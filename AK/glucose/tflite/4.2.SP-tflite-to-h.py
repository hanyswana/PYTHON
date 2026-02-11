import sys

# Read arguments from subprocess call
tflite_model_path, header_model_path = sys.argv[1], sys.argv[2]
array_name = 'tflite_model'  # Variable name for the array in the C header file

def tflite_to_c_header(tflite_model_path, header_model_path, array_name):
    # Read the TFLite model file
    with open(tflite_model_path, 'rb') as file:
        tflite_model = file.read()

    # Convert model to a comma-separated string of hex values
    hex_array = ', '.join(f'0x{b:02x}' for b in tflite_model)

    # Format the string into a C header file
    header_content = f"""#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

const unsigned char {array_name}[] = {{
    {hex_array}
}};

const unsigned int {array_name}_len = sizeof({array_name});

#endif // {array_name.upper()}_H
"""

    # Write the header content to the specified header file
    with open(header_model_path, 'w') as file:
        file.write(header_content)

tflite_to_c_header(tflite_model_path, header_model_path, array_name)
