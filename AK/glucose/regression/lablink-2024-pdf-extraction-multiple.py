# import os
# import re
# import PyPDF2
# import openpyxl
#
#
# def extract_pdf_data(pdf_path):
#     data = {}
#     with open(pdf_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#
#         # Normalize text by removing unwanted symbols
#         text = re.sub(r"\*+", "", text).strip()
#
#         # Extract patient details
#         data['Patient Name'] = re.search(r"Patient Name\s*:\s*(.+)", text).group(1) if re.search(r"Patient Name\s*:\s*(.+)", text) else ""
#         data['Age'] = re.search(r"Age\s*:\s*(\d+Y)", text).group(1) if re.search(r"Age\s*:\s*(\d+Y)", text) else ""
#         data['Sex'] = re.search(r"Sex\s*:\s*(\w+)", text).group(1) if re.search(r"Sex\s*:\s*(\w+)", text) else ""
#
#         # Define regex patterns for parameters
#         parameters = {
#             "Haemoglobin (g/dL)": r"Haemoglobin\s+([\d.]+)",
#             "Red cell count (10^12/L)": r"Red cell count\s+([\d.]+)",
#             "Haematocrit (PCV) (%)": r"Haematocrit \(PCV\)\s+([\d.]+)",
#             "MCV (fL)": r"MCV\s+([\d.]+)",
#             "MCH (pg)": r"MCH\s+([\d.]+)",
#             "MCHC (g/dL)": r"MCHC\s+([\d.]+)",
#             "RDW (%)": r"RDW\s+([\d.]+)",
#             "Platelet count (10^3/uL)": r"Platelet count\s+([\d.]+)",
#             "MPV (fL)": r"MPV\s+([\d.]+)",
#             "White blood cell count (10^3/uL)": r"White blood cell count\s+([\d.]+)",
#             "Neutrophil count (10^3/uL)": r"Neutrophil count\s+([\d.]+)",
#             "Lymphocyte count (10^3/uL)": r"Lymphocyte count\s+([\d.]+)",
#             "Eosinophil count (10^3/uL)": r"Eosinophil count\s+([\d.]+)",
#             "Monocyte count (10^3/uL)": r"Monocyte count\s+([\d.]+)",
#             "Basophil count (10^3/uL)": r"Basophil count\s+([\d.]+)",
#             "Neutrophil (%)": r"Neutrophil\s+([\d.]+) %",
#             "Lymphocyte (%)": r"Lymphocyte\s+([\d.]+) %",
#             "Eosinophil (%)": r"Eosinophil\s+([\d.]+) %",
#             "Monocyte (%)": r"Monocyte\s+([\d.]+) %",
#             "Basophil (%)": r"Basophil\s+([\d.]+) %",
#             "Neutrophil to lymphocyte ratio (<)": r"Neutrophil to lymphocyte ratio\s+([\d.]+)",
#             "Glucose (mmol/L)": r"Glucose\s+([\d.]+)\s+mmol/L",
#             "Total cholesterol (mmol/L)": r"Total cholesterol\s+([\d.]+)",
#             "Triglycerides (mmol/L)": r"Triglycerides\s+([\d.]+)",
#             "HDL cholesterol (mmol/L)": r"HDL cholesterol\s+([\d.]+)",
#             "Non HDL cholesterol (mmol/L)": r"Non HDL cholesterol\s+([\d.]+)",
#             "LDL cholesterol (mmol/L)": r"LDL cholesterol\s+([\d.]+)",
#             "Chol/HDL Chol": r"Chol/HDL Chol\s+([\d.]+)"
#         }
#
#         # Extract parameters using regex
#         for parameter, pattern in parameters.items():
#             match = re.search(pattern, text)
#             if match:
#                 data[parameter] = match.group(1)
#             else:
#                 data[parameter] = "N/A"  # Fill missing values with "N/A"
#                 print(f"Debug: Missing value for {parameter} in {pdf_path}")
#
#     return data
#
#
# def save_combined_to_excel(all_data, excel_path, headers):
#     workbook = openpyxl.Workbook()
#     sheet = workbook.active
#     sheet.title = "Lab Results"
#
#     # Write headers
#     sheet.append(headers)
#
#     # Write data
#     for data in all_data:
#         row = [data.get(header, "") for header in headers]
#         sheet.append(row)
#
#     # Save workbook
#     workbook.save(excel_path)
#     print(f"Data successfully saved to {excel_path}")
#
#
# # Process all PDFs in a folder
# folder_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/200-data-lablink-pdf/'
# excel_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/200-data-lablink-extracted.xlsx'
#
# headers = [
#     "Patient Name", "Age", "Sex", "Haemoglobin (g/dL)", "Red cell count (10^12/L)", "Haematocrit (PCV) (%)",
#     "MCV (fL)", "MCH (pg)", "MCHC (g/dL)", "RDW (%)", "Platelet count (10^3/uL)", "MPV (fL)",
#     "White blood cell count (10^3/uL)", "Neutrophil count (10^3/uL)", "Lymphocyte count (10^3/uL)",
#     "Eosinophil count (10^3/uL)", "Monocyte count (10^3/uL)", "Basophil count (10^3/uL)",
#     "Neutrophil (%)", "Lymphocyte (%)", "Eosinophil (%)", "Monocyte (%)", "Basophil (%)",
#     "Neutrophil to lymphocyte ratio (<)", "Glucose (mmol/L)", "Total cholesterol (mmol/L)",
#     "Triglycerides (mmol/L)", "HDL cholesterol (mmol/L)", "Non HDL cholesterol (mmol/L)",
#     "LDL cholesterol (mmol/L)", "Chol/HDL Chol"
# ]
#
# all_data = []
# for filename in os.listdir(folder_path):
#     if filename.endswith('.pdf'):
#         pdf_path = os.path.join(folder_path, filename)
#         print(f"Processing {pdf_path}...")
#         extracted_data = extract_pdf_data(pdf_path)
#         all_data.append(extracted_data)
#
# # Save the extracted data
# save_combined_to_excel(all_data, excel_path, headers)



import os
import re
import PyPDF2
import openpyxl


def extract_pdf_data(pdf_path):
    data = {}
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Remove Chinese characters and other non-Latin symbols
        text = re.sub(r"[^\x00-\x7F]+", " ", text).strip()

        # Normalize text by removing unwanted symbols
        text = re.sub(r"\*+", "", text).strip()

        # Extract patient details
        data['Full Patient Name'] = re.search(r"Patient Name\s*:\s*(.+)", text).group(1) if re.search(r"Patient Name\s*:\s*(.+)", text) else "N/A"
        data['Age'] = re.search(r"Age\s*:\s*(\d+Y)", text).group(1) if re.search(r"Age\s*:\s*(\d+Y)", text) else "N/A"
        data['Sex'] = re.search(r"Sex\s*:\s*(\w+)", text).group(1) if re.search(r"Sex\s*:\s*(\w+)", text) else "N/A"

        # Define regex patterns for parameters
        parameters = {
            "Haemoglobin (g/dL)": r"Haemoglobin\s+([\d.]+)",
            "Red cell count (10^12/L)": r"Red cell count\s+([\d.]+)",
            "Haematocrit (PCV) (%)": r"Haematocrit \(PCV\)\s+([\d.]+)",
            "MCV (fL)": r"MCV\s+([\d.]+)",
            "MCH (pg)": r"MCH\s+([\d.]+)",
            "MCHC (g/dL)": r"MCHC\s+([\d.]+)",
            "RDW (%)": r"RDW\s+([\d.]+)",
            "Platelet count (10^3/uL)": r"Platelet count\s+([\d.]+)",
            "MPV (fL)": r"MPV\s+([\d.]+)",
            "White blood cell count (10^3/uL)": r"White blood cell count\s+([\d.]+)",
            "Neutrophil count (10^3/uL)": r"Neutrophil count\s+([\d.]+)",
            "Lymphocyte count (10^3/uL)": r"Lymphocyte count\s+([\d.]+)",
            "Eosinophil count (10^3/uL)": r"Eosinophil count\s+([\d.]+)",
            "Monocyte count (10^3/uL)": r"Monocyte count\s+([\d.]+)",
            "Basophil count (10^3/uL)": r"Basophil count\s+([\d.]+)",
            "Neutrophil (%)": r"Neutrophil\s+([\d.]+) %",
            "Lymphocyte (%)": r"Lymphocyte\s+([\d.]+) %",
            "Eosinophil (%)": r"Eosinophil\s+([\d.]+) %",
            "Monocyte (%)": r"Monocyte\s+([\d.]+) %",
            "Basophil (%)": r"Basophil\s+([\d.]+) %",
            # "Neutrophil to lymphocyte ratio (<)": r"Neutrophil to lymphocyte ratio\s+([\d.]+)",
            "Neutrophil to lymphocyte ratio (<)": r"Neutrophil\s*to\s*lymphocyte\s*ratio\s*<?\s*([\d.]+)",
            "Glucose (mmol/L)": r"Glucose\s+([\d.]+)\s+mmol/L",
            "Total cholesterol (mmol/L)": r"Total cholesterol\s+([\d.]+)",
            "Triglycerides (mmol/L)": r"Triglycerides\s+([\d.]+)",
            "HDL cholesterol (mmol/L)": r"HDL cholesterol\s+([\d.]+)",
            "Non HDL cholesterol (mmol/L)": r"Non HDL cholesterol\s+([\d.]+)",
            "LDL cholesterol (mmol/L)": r"LDL cholesterol\s+([\d.]+)",
            # "Chol/HDL Chol": r"Chol/HDL Chol\s+([\d.]+)"
            "Chol/HDL Chol": r"Chol\s*/\s*HDL\s*Chol\s*([\d.]+)"
        }

        # Extract parameters using regex
        for parameter, pattern in parameters.items():
            match = re.search(pattern, text)
            if match:
                data[parameter] = match.group(1)
            else:
                data[parameter] = "N/A"  # Fill missing values with "N/A"
                print(f"Debug: Missing value for {parameter} in {pdf_path}")

    return data


def save_combined_to_excel(all_data, excel_path, headers):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Lab Results"

    # Write headers
    sheet.append(headers)

    # Write data
    for data in all_data:
        row = [data.get(header, "") for header in headers]
        sheet.append(row)

    # Save workbook
    workbook.save(excel_path)
    print(f"Data successfully saved to {excel_path}")


# Process all PDFs in a folder
folder_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/200-lablink-pdf/'
excel_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/lablink-2024-extraction/extracted-data/200-lablink-extracted.xlsx'

headers = [
    "Patient Name", "Age", "Sex", "Haemoglobin (g/dL)", "Red cell count (10^12/L)", "Haematocrit (PCV) (%)",
    "MCV (fL)", "MCH (pg)", "MCHC (g/dL)", "RDW (%)", "Platelet count (10^3/uL)", "MPV (fL)",
    "White blood cell count (10^3/uL)", "Neutrophil count (10^3/uL)", "Lymphocyte count (10^3/uL)",
    "Eosinophil count (10^3/uL)", "Monocyte count (10^3/uL)", "Basophil count (10^3/uL)",
    "Neutrophil (%)", "Lymphocyte (%)", "Eosinophil (%)", "Monocyte (%)", "Basophil (%)",
    "Neutrophil to lymphocyte ratio (<)", "Glucose (mmol/L)", "Total cholesterol (mmol/L)",
    "Triglycerides (mmol/L)", "HDL cholesterol (mmol/L)", "Non HDL cholesterol (mmol/L)",
    "LDL cholesterol (mmol/L)", "Chol/HDL Chol"
]

all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, filename)
        print(f"Processing {pdf_path}...")
        extracted_data = extract_pdf_data(pdf_path)
        all_data.append(extracted_data)

# Save the extracted data
save_combined_to_excel(all_data, excel_path, headers)
