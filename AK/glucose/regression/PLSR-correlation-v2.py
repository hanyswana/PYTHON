import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import os


def main():
    dir = '/home/apc-3/PycharmProjects/PythonProjectIV/SBH/SBH-PLSR'
    file = 'SBH-700-FS'
    file_path = f'{dir}/dataset/ori/csv/{file}.csv'
    model_filename = f'{dir}/model/{file}_PLSR_best_model.joblib'
    output_plot_path = f'{dir}/score-report/{file}_corr_plot.png'

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:]
    y = data.iloc[:, 2]
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    plsr_model = load(model_filename)

    y_pred = plsr_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, s=250, color='navy')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
    plt.title(f'PLSR Correlation of Honey Purity Levels for class FS (Fructose Syrup)\nSample Count: {len(y_test)}')
    plt.xlabel('Actual Honey Purity Level (%)')
    plt.ylabel('Predicted Honey Purity Level (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
