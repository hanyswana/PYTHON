import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os


def plot_regression_coefficients():
    dir = '/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR'
    file = 'SBH-700-DW'
    file_path = f'{dir}/dataset/ori/csv/{file}.csv'
    model_filename = f'{dir}/model/{file}_PLSR_best_model.joblib'
    output_plot_path = f'{dir}/score-report/{file}_coefficients_plot.png'

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    data = pd.read_csv(file_path)
    wavelength_columns = data.columns[3:]
    print(wavelength_columns)

    plsr_model = load(model_filename)
    coefficients = plsr_model.named_steps['plsr'].coef_.ravel()

    # Identify max and min coefficients
    max_idx = coefficients.argmax()
    min_idx = coefficients.argmin()
    max_wavelength = wavelength_columns[max_idx]
    min_wavelength = wavelength_columns[min_idx]
    max_value = coefficients[max_idx]
    min_value = coefficients[min_idx]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(wavelength_columns, coefficients, color='blue')
    plt.xticks(rotation=90)
    plt.title('Regression Coefficients per Wavelength Class DW (Distilled Water)')
    plt.xlabel('Wavelength')
    plt.ylabel('Coefficient Value')
    plt.grid(True)

    # Highlight and annotate max and min bars
    bars[max_idx].set_color('green')
    bars[min_idx].set_color('red')

    # Add annotation text above the plot
    textstr = (
        f'Highest Positive Coefficient: {max_wavelength} = {max_value:.4f}    |    '
        f'Highest Negative Coefficient: {min_wavelength} = {min_value:.4f}'
    )

    # Place annotation in the top-left inside the plot, below title
    plt.annotate(
        f'Highest + Coef: {max_wavelength} = {max_value:.1f}\n'
        f'Highest - Coef: {min_wavelength} = {min_value:.1f}',
        xy=(0.01, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )


    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_regression_coefficients()
