import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np


def save_evaluation_to_csv():
    train_data_path = f'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/dataset-2/PLS/br/pls-top-10-parquet/sbh-1/dataset-sbh-1_br_HW_top_10.parquet'
    data = pd.read_parquet(train_data_path)
    X_train = data.loc[:, '415 nm':]
    y_train = data.iloc[:, 2]

    model_path = f'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/model-2/Best_PLSR_model_sbh-1_br_HF_top_10.joblib'
    model = joblib.load(model_path)

    y_train_pred = model.predict(X_train)

    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2 = r2_score(y_train, y_train_pred)

    metrics_df = pd.DataFrame({
        'RMSE': [rmse],
        'R2': [r2]
    })

    metrics_df.to_csv('C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/score-2/SBH1_PLSR_confusion_metrics.csv', index=False)
    print("Evaluation metrics are saved to CSV file.")

if __name__ == "__main__":
    save_evaluation_to_csv()
