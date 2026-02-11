import sys
import pandas as pd

def extract_wavelength_indices(data_19_name, data_name, wavelength_10_header):
    """Extracts the indices of the selected wavelengths and saves them in a header file for Arduino."""

    # Load the full list of 19 wavelengths from CSV (starting from the third column)
    df_csv = pd.read_csv(data_19_name)
    full_wavelengths = list(df_csv.columns[2:])  # Start from third column

    # Load the selected 10 wavelengths from Parquet (also from the third column onward)
    df_parquet = pd.read_parquet(data_name)
    selected_wavelengths = list(df_parquet.columns[2:])  # Start from third column

    # Standardize formatting: remove spaces and ensure consistent string format
    full_wavelengths = [str(w).strip() for w in full_wavelengths]
    selected_wavelengths = [str(w).strip() for w in selected_wavelengths]

    # Find the correct indices of selected wavelengths in the full CSV list
    indices = []
    for wl in selected_wavelengths:
        if wl in full_wavelengths:
            indices.append(full_wavelengths.index(wl))  # No need for +1 since we start from col 2
        else:
            print(f"Error: Wavelength {wl} from Parquet not found in CSV!")
            sys.exit(1)

    # Write indices to a C++ header file
    header_content = f"""#ifndef WAVELENGTH_INDICES_H
#define WAVELENGTH_INDICES_H

const int WAVELENGTH_INDICES[10] = {{{", ".join(map(str, indices))}}};

#endif // WAVELENGTH_INDICES_H
"""

    with open(wavelength_10_header, "w") as f:
        f.write(header_content)

    print(f"Wavelength indices saved to: {wavelength_10_header}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract-wavelength.py <data_19_name> <data_name> <wavelength_10_header>")
        sys.exit(1)

    data_19_name = sys.argv[1]
    data_name = sys.argv[2]
    wavelength_10_header = sys.argv[3]

    extract_wavelength_indices(data_19_name, data_name, wavelength_10_header)
