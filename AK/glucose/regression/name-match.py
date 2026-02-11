import pandas as pd

# Load the data from the provided Excel files
label_file_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/label.xlsx'
data_file_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/200-data-lablink-extracted.xlsx'

# Read the Label and Data files
labels_df = pd.read_excel(label_file_path)
data_df = pd.read_excel(data_file_path)

# Normalize simplified patient names and full patient names for case and space insensitivity
labels_df['Simplified Patient Name'] = labels_df['Simplified Patient Name'].str.strip().str.upper()
data_df['Normalized Full Name'] = data_df['Full Patient Name'].str.strip().str.upper()

# Function to extract simplified names from full names considering different positions
def find_simplified_name(full_name, simplified_name):
    parts = full_name.split()
    # Check if the simplified name is part of the full name (any part)
    for part in parts:
        if simplified_name in part:
            return True
    return False

# Match the simplified names to full patient names
def match_names(simplified_name):
    for full_name in data_df['Full Patient Name']:
        if find_simplified_name(full_name.upper(), simplified_name):
            return full_name
    return 'Unmatched'

labels_df['Matched Full Name'] = labels_df['Simplified Patient Name'].apply(match_names)

# Save the results to an Excel file
output_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/matched_name.xlsx'
labels_df.to_excel(output_path, index=False)

print(f"Matching complete. Results saved to {output_path}.")
