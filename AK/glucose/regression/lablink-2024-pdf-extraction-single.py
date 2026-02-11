import re
import PyPDF2
import openpyxl

# Define a function to extract text from the PDF
def extract_pdf_data(pdf_path):
    data = {}
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Extract patient details
        data['Patient Name'] = re.search(r"Patient Name\s*:\s*(.+)", text).group(1)
        data['Age'] = re.search(r"Age\s*:\s*(\d+Y)", text).group(1)
        data['Sex'] = re.search(r"Sex\s*:\s*(\w+)", text).group(1)

        # Extract Full Blood Count (FBC)
        fbc_match = re.search(r"FULL BLOOD COUNT \(GP1A\)(.*?)-- White blood cell differential count --", text, re.S)
        if fbc_match:
            fbc_section = fbc_match.group(1).strip()
            for line in fbc_section.splitlines():
                match = re.match(r"(.+?)\s+([\d.]+)\s*([\w/%<>=.-]+)?", line)
                if match:
                    parameter, value, unit_or_range = match.groups()
                    if "1012/L" in unit_or_range:
                        unit_or_range = unit_or_range.replace("1012/L", "10^12/L")
                    if "103/uL" in unit_or_range or "10^3/uL" in unit_or_range:
                        unit_or_range = "10^3/uL"
                    # Update parameter name to include the unit and save only the value
                    if unit_or_range:
                        parameter = f"{parameter.strip()} ({unit_or_range.strip()})"
                    value = value.rstrip("<").strip()
                    data[parameter.strip()] = value

        # Extract White Blood Cell Differential Count
        wbc_match = re.search(r"-- White blood cell differential count --(.*?)RANDOM BLOOD SUGAR \(RBS\)", text, re.S)
        if wbc_match:
            wbc_section = wbc_match.group(1).strip()
            for line in wbc_section.splitlines():
                match = re.match(r"(.+?)\s+([\d.]+)\s*([\w/%<>=.-]+)?", line)
                if match:
                    parameter, value, unit_or_range = match.groups()
                    if "1012/L" in unit_or_range:
                        unit_or_range = unit_or_range.replace("1012/L", "10^12/L")
                    if "103/uL" in unit_or_range or "10^3/uL" in unit_or_range:
                        unit_or_range = "10^3/uL"
                    # Update parameter name to include the unit and save only the value
                    if unit_or_range:
                        parameter = f"{parameter.strip()} ({unit_or_range.strip()})"
                    value = value.rstrip("<").strip()
                    data[parameter.strip()] = value

        # Extract RANDOM BLOOD SUGAR (RBS)
        rbs_match = re.search(r"RANDOM BLOOD SUGAR \(RBS\)(.*?)Fasting blood glucose interpretation:", text, re.S)
        if rbs_match:
            rbs_section = rbs_match.group(1).strip()
            for line in rbs_section.splitlines():
                if "Glucose" in line:
                    match = re.match(r"Glucose\s+([\d.]+)\s+mmol/L", line)
                    if match:
                        value = match.group(1).rstrip("<").strip()  # Correct the '<' issue
                        data["Glucose (mmol/L)"] = value

        # Extract Lipid Profile
        lipid_match = re.search(r"LIPID PROFILE \(GP24\)(.*?)Reference:", text, re.S)
        if lipid_match:
            lipid_section = lipid_match.group(1).strip()
            for line in lipid_section.splitlines():
                match = re.match(r"(.+?)\s+([\d.]+)\s*([\w/%<>=.-]+)?", line)
                if match:
                    parameter, value, unit_or_range = match.groups()
                    if "1012/L" in unit_or_range:
                        unit_or_range = unit_or_range.replace("1012/L", "10^12/L")
                    # Update parameter name to include the unit and save only the value
                    if unit_or_range:
                        parameter = f"{parameter.strip()} ({unit_or_range.strip()})"
                    value = value.rstrip("<").strip()
                    data[parameter.strip()] = value

    # Explicit correction for specific keys like "Neutrophil to lymphocyte ratio"
    for key in data.keys():
        if "Neutrophil to lymphocyte ratio" in key:
            data[key] = data[key].replace("<", "").strip()

    # Filter out unwanted parameters
    unwanted_parameters = [
        "Low CV Risk",
        "Moderate CV Risk",
        "Morderate CV Risk",  # Explicitly add misspelling here
        "High CV Risk",
        "Very High CV Risk",
        "Chol/HDL Chol",
        "GFR",
        "Low and Moderate CV Risk"
    ]

    # Strict filtering using fuzzy and substring matching
    filtered_data = {}
    for key, value in data.items():
        # Check if any unwanted substring is part of the parameter key (case insensitive)
        if not any(unwanted.lower() in key.lower() for unwanted in unwanted_parameters):
            filtered_data[key] = value
        else:
            print(f"Filtered out: {key}")  # Debug log for verification

    return filtered_data

# Function to save transposed data into Excel
def save_to_excel(data, excel_path):
    # Create a workbook and select the active worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Lab Results"

    # Write the transposed data: Keys as headers in the first row, values in the second row
    sheet.append(list(data.keys()))  # Headers
    sheet.append(list(data.values()))  # Values

    # Save the workbook
    workbook.save(excel_path)
    print(f"Data successfully saved to {excel_path}")

# Main logic
pdf_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/0124042661.pdf'
excel_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/lab_results_transposed_with_units.xlsx'

# Extract data from PDF
extracted_data = extract_pdf_data(pdf_path)

# Save the extracted data to Excel
save_to_excel(extracted_data, excel_path)



