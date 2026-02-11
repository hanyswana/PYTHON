import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
from scipy import sparse
import numpy as np


class BaselineRemover(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = self.remove_baseline(X.T).T
        return X

    def remove_baseline(self, spectra):
        return spectra - spectra.mean(axis=0)

    def _more_tags(self):
        return {'allow_nan': True}


# def apply_moving_average_smoothing(spectral_data, window_size=25):
#     return spectral_data.apply(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'))


def predict_single_instance():
    model_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/model-2/Best_PLSR_model_sbh-1_br_HF_top_10.joblib'
    model = joblib.load(model_path)

    new_data_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/dataset-2/PLS/br/pls-top-10-csv/dataset-sbh-1_br_HF_top_10.csv'
    new_data = pd.read_csv(new_data_path)

    # if the new_data has been applied with preprocess
    # single_instance = new_data.loc[0, '415 nm':].values.reshape(1, -1)       # one row in data
    # single_instance = new_data.loc[:, '415 nm':]     # all rows in data

    # smoothed_data = apply_moving_average_smoothing(new_data.loc[:, '415 nm':])

    smoothed_data = new_data.loc[:, '415 nm':]
    # baseline_remover = BaselineRemover()
    # preprocessed_data = baseline_remover.transform(smoothed_data)
    # preprocessed_data_df = pd.DataFrame(preprocessed_data, columns=new_data.columns[4:])
    # print(preprocessed_data_df)

    # single_instance = preprocessed_data_df.iloc[35].values.reshape(1, -1)
    single_instance = smoothed_data.iloc[0].values.reshape(1, -1)
    # print(single_instance)

    prediction = model.predict(single_instance)
    rounded_prediction = round(prediction[0], 0)
    print(f"Honey: {rounded_prediction} °%")


if __name__ == "__main__":
    predict_single_instance()
