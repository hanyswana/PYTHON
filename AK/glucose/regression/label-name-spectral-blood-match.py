import pandas as pd

# Load the uploaded files
data_lablink = pd.ExcelFile('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/200-lablink-extracted.xlsx')
workbook = pd.ExcelFile('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/Lablink-Nov-2024-workbook.xlsx')
patient_name_label = pd.ExcelFile('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/label_name.xlsx')

# Extract relevant sheets for merging
lablink_data = data_lablink.parse(data_lablink.sheet_names[0])  # First sheet
workbook_data = workbook.parse(workbook.sheet_names[0])  # First sheet
patient_data = patient_name_label.parse(patient_name_label.sheet_names[0])  # First sheet

# Merge workbook data with patient data on the 'Label' column
merged_spectral = pd.merge(workbook_data, patient_data[['Label', 'Full Patient Name']], on='Label', how='left')

# Merge the result with lablink data on 'Full Patient Name'
final_merged_data = pd.merge(merged_spectral, lablink_data, on='Full Patient Name', how='left')

# Reorganize columns: move spectral data (415 nm to 940 nm) to the end
spectral_columns = [col for col in final_merged_data.columns if 'nm' in col]
non_spectral_columns = [col for col in final_merged_data.columns if col not in spectral_columns]

# Reorder columns
final_merged_data = final_merged_data[non_spectral_columns + spectral_columns]

# Save the final dataset to a file
final_merged_data.to_excel('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/200-lablink-spectral-workbook.xlsx', index=False)

# Display message
print("The final dataset has been saved as '200-lablink-spectral.xlsx'.")
