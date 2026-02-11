import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
import re
from scipy import sparse

'''
Log v 0.2
To use the code do edit:
1.feature starting column on line 96(raw) and 136 (processed)
2.input folder (line 150) and output folder (line 151)
'''


# Define the SNV function
def snv(spectra):
    mean_corrected = spectra - np.mean(spectra, axis=1, keepdims=True)
    return mean_corrected / np.std(mean_corrected, axis=1, keepdims=True)


# Define the Baseline Remover
class BaselineRemover(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        return X - np.mean(X, axis=1, keepdims=True)

    def _more_tags(self):
        return {'allow_nan': True}


# Define the Savitzky-Golay filter function
def mirror_pad(y, half_window):
    left_pad = y[1:half_window + 1][::-1]
    right_pad = y[-half_window - 1:-1][::-1]
    return np.concatenate((left_pad, y, right_pad))


def SavGol(y, window_size, poly_order, deriv_order=0):
    window_size = int(window_size)
    poly_order = int(poly_order)
    if window_size % 2 != 1 or window_size < 1:
        raise ValueError("window_size must be a positive odd number")
    if window_size < poly_order + 2:
        raise ValueError("window_size is too small for the polynomial order")
    half_window = (window_size - 1) // 2
    b = np.array([[k ** i for i in range(poly_order + 1)] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b)[deriv_order] * (np.math.factorial(deriv_order))
    y_padded = mirror_pad(y, half_window)
    return np.convolve(m[::-1], y_padded, mode='valid')


# Define preprocessing functions
def apply_snv(X):
    return pd.DataFrame(snv(X.values), columns=X.columns, index=X.index)


def apply_norm_euc(X):
    normalizer = Normalizer(norm='l2')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)


def apply_norm_manh(X):
    normalizer = Normalizer(norm='l1')
    return pd.DataFrame(normalizer.fit_transform(X), columns=X.columns, index=X.index)


def apply_baseline(X):
    baseline_remover = BaselineRemover()
    return pd.DataFrame(baseline_remover.fit_transform(X), columns=X.columns, index=X.index)


def apply_first_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=1),
                        columns=X.columns, index=X.index)


def apply_second_derivative(X):
    return pd.DataFrame(np.apply_along_axis(SavGol, 1, X.values, window_size=5, poly_order=2, deriv_order=2),
                        columns=X.columns, index=X.index)


# Define the main processing function
def process_file(file_path, output_folder):
    df = pd.read_csv(file_path)
    X = df.iloc[:, 4:] # The feature start column

    # Define the 24 specific combinations
    combinations = [
        ('RAW', lambda x: x),
        ('Baseline', apply_baseline),
        ('Norm_Manh', apply_norm_manh),
        ('Norm_Euc', apply_norm_euc),
        ('SNV', apply_snv),
        ('SNV_Norm_Euc', lambda x: apply_norm_euc(apply_snv(x))),
        ('SNV_Norm_Manh', lambda x: apply_norm_manh(apply_snv(x))),
        ('SNV_Baseline', lambda x: apply_baseline(apply_snv(x))),
        ('SNV_Norm_Euc_Baseline', lambda x: apply_baseline(apply_norm_euc(apply_snv(x)))),
        ('SNV_Norm_Manh_Baseline', lambda x: apply_baseline(apply_norm_manh(apply_snv(x)))),
        ('SNV_Norm_Manh_1st_Deriv', lambda x: apply_first_derivative(apply_norm_manh(apply_snv(x)))),
        ('SNV_Norm_Manh_1st_Deriv_Baseline',
         lambda x: apply_baseline(apply_first_derivative(apply_norm_manh(apply_snv(x))))),
        ('SNV_Norm_Manh_2nd_Deriv', lambda x: apply_second_derivative(apply_norm_manh(apply_snv(x)))),
        ('SNV_Norm_Manh_2nd_Deriv_Baseline',
         lambda x: apply_baseline(apply_second_derivative(apply_norm_manh(apply_snv(x))))),
        ('Norm_Manh_SNV', lambda x: apply_snv(apply_norm_manh(x))),
        ('Norm_Manh_SNV_1st_Deriv', lambda x: apply_first_derivative(apply_snv(apply_norm_manh(x)))),
        ('Norm_Manh_SNV_1st_Deriv_Baseline',
         lambda x: apply_baseline(apply_first_derivative(apply_snv(apply_norm_manh(x))))),
        ('Norm_Manh_SNV_2nd_Deriv', lambda x: apply_second_derivative(apply_snv(apply_norm_manh(x)))),
        ('Norm_Manh_SNV_2nd_Deriv_Baseline',
         lambda x: apply_baseline(apply_second_derivative(apply_snv(apply_norm_manh(x))))),
        ('Norm_Euc_SNV', lambda x: apply_snv(apply_norm_euc(x))),
        ('Norm_Euc_SNV_1st_Deriv', lambda x: apply_first_derivative(apply_snv(apply_norm_euc(x)))),
        ('Norm_Euc_SNV_1st_Deriv_Baseline',
         lambda x: apply_baseline(apply_first_derivative(apply_snv(apply_norm_euc(x))))),
        ('Norm_Euc_SNV_2nd_Deriv', lambda x: apply_second_derivative(apply_snv(apply_norm_euc(x)))),
        ('Norm_Euc_SNV_2nd_Deriv_Baseline',
         lambda x: apply_baseline(apply_second_derivative(apply_snv(apply_norm_euc(x)))))
    ]

    for combo_name, func in combinations:
        X_processed = func(X)

        # Concatenate processed data with original dataframe
        df_processed = pd.concat([df.iloc[:, :4], X_processed], axis=1)

        # Create output file name
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(output_folder, f"{name}_{combo_name}{ext}")

        # Save processed data to csv
        df_processed.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


# Add new function to process small values
def check_small_values(output_folder):
    """Process all CSV files in output folder to convert very small values to 0"""
    print("\nStarting small value check...")
    processed_files = 0
    modified_files = 0

    # Get all CSV files
    csv_files = glob.glob(os.path.join(output_folder, '*.csv'))

    for file_path in csv_files:
        processed_files += 1
        print(f"Processing file {processed_files}/{len(csv_files)}: {os.path.basename(file_path)}")

        # Read CSV file
        df = pd.read_csv(file_path)
        original_df = df.copy()

        # Get numerical columns (skip first four columns)
        numerical_cols = df.columns[4:]

        modified_columns = []

        # Convert very small values to 0
        for col in numerical_cols:
            original_col = df[col].copy()

            # Convert to string and check for actual decimal places
            df[col] = df[col].apply(lambda x: check_decimal_zeros(x))

            # Check if this column was modified
            if not df[col].equals(original_col):
                modified_columns.append(col)

        # Save if changes were made
        if modified_columns:  # Only save if there were modifications
            df.to_csv(file_path, index=False)
            modified_files += 1
            print(f"Modified and saved: {os.path.basename(file_path)}")
            print(f"Modified columns: {', '.join(modified_columns)}")

    print(f"\nSmall value check completed:")
    print(f"Total files processed: {processed_files}")
    print(f"Files modified: {modified_files}")


def check_decimal_zeros(x):
    """
    Check if a number effectively has 5 or more consecutive zeros after the decimal point,
    handling both regular decimal notation and scientific notation.

    Examples:
        0.000002135245 -> 0  (converts to 0 because it has 5 zeros after decimal)
        1.000001345 -> 1.000001345 (preserved because whole number is not 0)
        -3.46944695195361E-18 -> 0 (converts to 0 because in decimal form it would have 18 zeros)
        2e-7 -> 0 (converts to 0 because it represents 0.0000002)
        1e-4 -> 0.0001 (preserved because it only has 4 zeros)
    """
    if not isinstance(x, (int, float)):
        return x

    # Convert to string and handle scientific notation
    str_val = str(x)

    # If it's in scientific notation
    if 'e' in str_val.lower():
        base, exponent = str_val.lower().split('e')
        exp = int(exponent)

        # If the exponent is negative and <= -5, it means we have at least 5 zeros after decimal
        if exp <= -5:
            return 0

        # For positive exponents or small negative exponents, use original value
        return x

    # For non-scientific notation, use the original logic
    if '.' not in str_val:
        return x

    # Split on decimal point and check the decimal part
    whole, decimal = str_val.split('.')

    # Only proceed if the whole number part is 0 or -0
    if whole not in ['0', '-0']:
        return x

    # Remove trailing zeros from the decimal part
    decimal = decimal.rstrip('0')

    # Check if the decimal part starts with 5 or more zeros
    if re.match(r'^0{5}', decimal):
        return 0

    return x


# Main execution
if __name__ == "__main__":
    input_folder = '/home/apc-3/PycharmProjects/PythonProjectAK/DATA-diagnostic/dataset-lablink-alty/alty-raw'
    output_folder = '/home/apc-3/PycharmProjects/PythonProjectAK/DATA-diagnostic/dataset-lablink-alty/alty-24'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # Process each CSV file
    for file_path in csv_files:
        process_file(file_path, output_folder)

    print("Processing completed for all files.")

    # Check and process small values
    check_small_values(output_folder)

    print("All processing completed successfully.")
