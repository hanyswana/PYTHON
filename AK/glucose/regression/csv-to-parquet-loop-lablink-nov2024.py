import pandas as pd
from pathlib import Path

path_dir = '/home/apc-3/PycharmProjects/PythonProjectAK/lablink-nov2024-1k-glucose/dataset-1k-glucose/new/dataset'
csv_dir = Path(f'{path_dir}/csv-10-PLS')
parquet_dir = Path(f'{path_dir}/parquet-10-PLS')

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


