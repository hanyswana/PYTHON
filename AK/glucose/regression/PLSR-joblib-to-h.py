import joblib

# Define the suffix
suffix = "_fs"

# Load the joblib model
file = 'SBH-700-FS_PLSR_best_model'
model_path = f'/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR/model/{file}.joblib'
pipeline = joblib.load(model_path)

# Extract the scaler and PLSR model from the pipeline
scaler = pipeline.named_steps['scaler']
plsr_model = pipeline.named_steps['plsr']

# Get the coefficients and intercept from the PLSR model
# coefficients = plsr_model.coef_
coefficients = plsr_model.coef_.reshape(-1)  # Ensures flat 1D array in the correct direction
intercept = plsr_model.intercept_

# Get the means and standard deviations from the StandardScaler
means = scaler.mean_
std_devs = scaler.scale_
n_features = coefficients.shape[0]

# Start constructing the header file content
header_content = f'#ifndef PLSR_MODEL{suffix.upper()}\n#define PLSR_MODEL{suffix.upper()}\n\n'

header_content += f'#define N_FEATURES{suffix} {n_features}\n\n'

# Add means and standard deviations to the header file with the suffix
# header_content += f'float means{suffix}[] = ' + '{' + ', '.join(map(str, means)) + '};\n'
header_content += f'float means{suffix}[] = ' + '{' + ', '.join(f'{x:.10f}' for x in means) + '};\n'
# header_content += f'float std_devs{suffix}[] = ' + '{' + ', '.join(map(str, std_devs)) + '};\n\n'
header_content += f'float std_devs{suffix}[] = ' + '{' + ', '.join(f'{x:.10f}' for x in std_devs) + '};\n\n'

# Add the coefficients and intercept with the suffix
# header_content += f'float coefficients{suffix}[] = ' + '{' + ', '.join(map(str, coefficients.flatten())) + '};\n'
header_content += f'float coefficients{suffix}[] = ' + '{' + ', '.join(f'{x:.10f}' for x in coefficients.flatten()) + '};\n'
# header_content += f'float intercept{suffix} = ' + str(intercept[0]) + ';\n\n'  # Correct intercept definition
header_content += f'float intercept{suffix} = ' + f'{intercept[0]:.10f}' + ';\n\n'  # Correct intercept definition

# Add the standardization function with the suffix
header_content += f'void standardize_input{suffix}(float input[], int length) {{\n'
header_content += '  for (int i = 0; i < length; ++i) {\n'
header_content += f'    input[i] = (input[i] - means{suffix}[i]) / std_devs{suffix}[i];\n'
header_content += '  }\n'
header_content += '}\n\n'

# Add the predict function with the suffix
header_content += f'float predict_plsr{suffix}(float input[]) {{\n'
header_content += f'  // First, standardize the input data using standardize_input{suffix}\n'
# header_content += f'  standardize_input{suffix}(input, 10);\n\n'
header_content += f'  standardize_input{suffix}(input, N_FEATURES{suffix});\n\n'
header_content += '  float result = 0.0;\n'
header_content += '  // Use all coefficients for prediction\n'
# header_content += f'  for (int i = 0; i < 10; i++) {{\n'
header_content += f'  for (int i = 0; i < N_FEATURES{suffix}; i++) {{\n'
header_content += f'    result += input[i] * coefficients{suffix}[i];\n'
header_content += '  }\n'
header_content += f'  return result + intercept{suffix};\n'
header_content += '}\n\n'

# End the header file
header_content += '#endif'

# Define the path for saving the header file
header_file_path = f'/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR/model-converted/{file}.h'

# Save the header file
with open(header_file_path, 'w') as file:
    file.write(header_content)

print("Header file created:", header_file_path)

print("Coefficients shape:", plsr_model.coef_.shape)
print("Intercept shape:", plsr_model.intercept_.shape)
print("Coefficients (first 5):", plsr_model.coef_.flatten()[:5])
print("Means (first 5):", scaler.mean_[:5])
print("Stds (first 5):", scaler.scale_[:5])

