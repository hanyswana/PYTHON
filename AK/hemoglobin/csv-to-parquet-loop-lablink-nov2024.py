import pandas as pd
from pathlib import Path

path_dir = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-nov2024-1k-hb/dataset-1k-hb/pls'
csv_dir = Path(f'{path_dir}/dataset-1k-pls-top10-csv')
parquet_dir = Path(f'{path_dir}/dataset-1k-pls-top10-parquet')

parquet_dir.mkdir(parents=True, exist_ok=True)

for csv_file in csv_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)

    # for col in df.columns:
    #     df[col] = df[col].astype('float32')

    # Convert only numeric columns to float32
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].astype('float32')

    new_file_path = parquet_dir / (csv_file.stem + '.parquet')

    df.to_parquet(new_file_path, index=False)

    print(f"Conversion complete. The modified DataFrame has been saved to {new_file_path}.")


# import pandas as pd
# from pathlib import Path
#
# # Define source and target directories
# csv_dir = Path('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/dataset/PLS/br/pls-top-10-csv')
# parquet_base_dir = Path('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/dataset/PLS/br/pls-top-10-parquet')
#
# # Ensure the base directory exists
# parquet_base_dir.mkdir(parents=True, exist_ok=True)
#
# # Iterate over each CSV file in the source directory
# for csv_file in csv_dir.glob('*.csv'):
#     df = pd.read_csv(csv_file)
#
#     # Convert only numeric columns to float32
#     for col in df.select_dtypes(include='number').columns:
#         df[col] = df[col].astype('float32')
#
#     # Create a unique folder for each Parquet file based on the CSV file name
#     parquet_dir = parquet_base_dir / csv_file.stem
#     parquet_dir.mkdir(parents=True, exist_ok=True)
#
#     # Save the Parquet file in its designated folder
#     new_file_path = parquet_dir / (csv_file.stem + '.parquet')
#     df.to_parquet(new_file_path, index=False)
#
#     print(f"Conversion complete. The modified DataFrame has been saved to {new_file_path}.")
