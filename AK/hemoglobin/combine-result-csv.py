import os
import pandas as pd

# Define the folder path containing the CSV files
folder_path = '/home/admin-3/PycharmProjects/PythonProjectAK-REVA/1k-lablink-L1L2-dropout/1k-hb/1k-hb-all-model-result'  # Replace with the path to your folder
output_path = '/home/admin-3/PycharmProjects/PythonProjectAK-REVA/1k-lablink-L1L2-dropout/1k-hb/1k-hb-all-model-result/combined-results-hb-1k-nonstratified.csv'  # Path to save the combined CSV file

# List all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Load each CSV file and append to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to: {output_path}")
