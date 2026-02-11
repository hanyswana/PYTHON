import tensorflow as tf

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# If no GPU is found, print an error message
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("⚠️ No GPU detected! Training may be slow.")
else:
    print("✅ GPU is detected and ready for use! GPU :")
    print(tf.config.list_physical_devices('GPU'))  # Should list at least one GPU

import os
import logging
import datetime
import csv
import pandas as pd
import numpy as np
import prefect
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
# from autokeras import StructuredDataRegressor
from prefect import flow, task, exceptions
# from prefect.tasks import task_input_hash
# from datetime import timedelta
from keras.callbacks import EarlyStopping, TensorBoard
import keras.callbacks
import json
import hashlib
import autokeras as ak
# from autokeras import Node, Input, StructuredDataInput
# from autokeras.engine import block as auto_model
from tensorflow.keras import regularizers
from autokeras import AutoModel
import psutil
import sys


"""Ver 1.0.2 Log
- add function to stop and re run the code when the ram usage hit 99%.
it will re run 3 times before the code stop

"""


class RAMUsageCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=98):
        super().__init__()
        self.threshold = threshold
        self.high_ram_usage = False

    def on_epoch_end(self, epoch, logs=None):
        ram_percent = psutil.virtual_memory().percent
        if ram_percent >= self.threshold:
            self.high_ram_usage = True
            logging.warning(f"RAM usage hit {ram_percent}%. Stopping training after this trial.")


# Base folder location
BASE_FOLDER = '/home/apc-3/PycharmProjects/PythonProjectAK/1k-lablink-L1L2-dropout/1k-hb/stratify-v2/2-modi-dropout-regu'

# Configure logging
logging.basicConfig(level=logging.INFO, filename=f'{BASE_FOLDER}/experiment_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File locations
DATASET_FOLDER = os.path.join(BASE_FOLDER, 'dataset/run/dataset-1k-hb-pls-top10-parquet-1-2-1')
TB_FOLDER = os.path.join(BASE_FOLDER, 'lablink-1k-SAFE (copy)/batch-size-16-4070ti-prefect/tb-topLV-batch-size')
CKPT_FOLDER = os.path.join(BASE_FOLDER, 'checkpoint')
MODEL_SAVE_FOLDER = os.path.join(BASE_FOLDER, '1k-hb-all-best-model')
HYPERPARAMETER_FOLDER = os.path.join(BASE_FOLDER, '1k-hb-all-model-hyperparameter')
RESULT_FOLDER = os.path.join(BASE_FOLDER, '1k-hb-all-model-result')
VALIDATION_FOLDER = os.path.join(BASE_FOLDER, 'dataset/lablink_test_data/output')


# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logging.error(e)


class OOMCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.oom_count = 0

    def on_epoch_end(self, epoch, logs=None):
        try:
            super().on_epoch_end(epoch, logs)
        except tf.errors.ResourceExhaustedError:
            raise prefect.exceptions.Retry("OOM error occurred. Retrying...")


class NaNCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key.endswith('loss') or key.endswith('mean_squared_error'):
                if tf.math.is_nan(value):
                    logs[key] = 10  # Replace NaN loss or metric with 10
                    logging.warning(f"Epoch {epoch}: Encountered NaN in {key}. Setting it to 10.")
        super().on_epoch_end(epoch, logs)


class ThreadCreationErrorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            super().on_epoch_end(epoch, logs)
        except tf.errors.InternalError as e:
            if "Thread tf_data_private_threadpool creation via pthread_create() failed" in str(e):
                raise RuntimeError("Thread creation error occurred.")




def save_hyperparameters(hyperparameters, hyperparameter_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(hyperparameter_file), exist_ok=True)
    # Save the hyperparameters with indentation for readability
    with open(hyperparameter_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)


def load_hyperparameters(hyperparameter_file):
    if not os.path.exists(hyperparameter_file) or os.path.getsize(hyperparameter_file) == 0:
        return {}
    with open(hyperparameter_file, 'r') as f:
        return json.load(f)


def get_hyperparameter_hash(hyperparameters):
    return hashlib.md5(json.dumps(hyperparameters, sort_keys=True).encode()).hexdigest()


class CustomStructuredDataRegressor(ak.StructuredDataRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self, hp):
        input_node = ak.StructuredDataInput()
        output_node = input_node

        output_node = ak.DenseBlock(
            num_layers=hp.Int('num_layers', min_value=1, max_value=10),
            num_units=hp.Int('units', min_value=4, max_value=128, step=8),
            dropout=hp.Float('dropout', min_value=0.0, max_value=0.2, step=0.05),
            use_batchnorm=False,
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1', min_value=1e-6, max_value=1e-5, sampling='log'),
                l2=hp.Float('l2', min_value=1e-6, max_value=1e-5, sampling='log')
            ),
            activation='relu'  # Add ReLU activation function
        )(output_node)

        output_node = ak.RegressionHead()(output_node)

        return ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True)


#################################


# @flow(retries=5, retry_delay_seconds=1800)
def train_and_predict(dataset):
    try:
        file = os.path.join(DATASET_FOLDER, dataset)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]

        # Dataset-specific paths
        model_save_folder = os.path.join(MODEL_SAVE_FOLDER, dataset_name)
        result_file = os.path.join(RESULT_FOLDER, f'results-all-datasets-1k-hb-stratify.csv')
        hyperparameter_file = os.path.join(HYPERPARAMETER_FOLDER, f'{dataset_name}_hyperparameter.json')
        tuning_folder = os.path.join(BASE_FOLDER, '1k-hb-all-tuning')
        validation_output_file = os.path.join(VALIDATION_FOLDER, f'validation_output_{dataset_name}.csv')

        os.makedirs(model_save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        os.makedirs(os.path.dirname(hyperparameter_file), exist_ok=True)
        os.makedirs(tuning_folder, exist_ok=True)
        os.makedirs(VALIDATION_FOLDER, exist_ok=True)

        df = pd.read_parquet(file, engine="pyarrow")
        logging.info(f"Loaded file: {file}")

        # Get the column names from the training data
        train_columns = df.columns.tolist()
        feature_columns = [col for col in train_columns if col not in ['ID', 'Haemoglobin (g/dL)']]

        X = df[feature_columns]
        y = df['Haemoglobin (g/dL)']


        # Stratify and balance the data by ensuring equal samples per bin (balance bins)
        # Define bins and labels
        bins = [0, 13, 16, float('inf')]  # Bins: 13 g/dL and below, 13-16 g/dL, 16 g/dL and above
        bin_labels = ['<=13', '13-16', '>=16']  # Corresponding labels

        # Add bin column to the dataframe
        df['Haemoglobin_bin'] = pd.cut(
            df['Haemoglobin (g/dL)'], bins=bins, labels=bin_labels, include_lowest=True
        )

        # Display the distribution of bins
        print(df['Haemoglobin_bin'].value_counts())

        # Split the data using stratification based on the bins
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=df['Haemoglobin_bin'], random_state=42)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=df.loc[X_temp.index, 'Haemoglobin_bin'], random_state=42)


        # Split data: 70% train, 15% validation, 15% test
        # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        # X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_validation.shape}, Test data shape: {X_test.shape}")

        # Dynamically create the hyperparameter file path
        hyperparameter_file = os.path.join(HYPERPARAMETER_FOLDER, f'{dataset_name}_hyperparameter.json')

        previous_hyperparameters = load_hyperparameters(hyperparameter_file)

        max_trials = 100

        hyperparameters = {
            'max_trials': max_trials,
            'project_name': f'{BASE_FOLDER}/1k-hb-all-tuning/hyperparameter-tuning-trials/{dataset_name}-tuning',
            'output_dim': 1,
            'tuner': 'greedy'
        }

        hyperparameter_hash = get_hyperparameter_hash(hyperparameters)
        if hyperparameter_hash in previous_hyperparameters:
            logging.info(f"Skipping previously used hyperparameters: {hyperparameters}")
            return

        # K-Fold Cross Validation
        cv_r2_scores = []
        cv_mse_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []  # To store results for each fold

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Proceed with the current hyperparameter setting
            reg = CustomStructuredDataRegressor(
                max_trials=max_trials,
                overwrite=True,
                loss='mean_squared_error',
                project_name=f'{BASE_FOLDER}/1k-hb-all-tuning/current/{dataset_name}-tuning',
                tuner='greedy'
            )

            nan_callback = NaNCallback()
            oom_callback = OOMCallback()
            thread_creation_error_callback = ThreadCreationErrorCallback()
            ram_usage_callback = RAMUsageCallback(threshold=98)

            reg.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), batch_size=4, callbacks=[nan_callback, oom_callback, thread_creation_error_callback, ram_usage_callback])
            y_pred = reg.predict(X_val_fold)

            # Calculate MSE and R2 for this fold
            fold_mse = mean_squared_error(y_val_fold, y_pred)
            fold_r2 = r2_score(y_val_fold, y_pred)

            # Log and store results
            logging.info(f"Fold {fold} MSE: {fold_mse}, R2: {fold_r2}")
            fold_results.append((fold, fold_mse, fold_r2))

            cv_mse_scores.append(fold_mse)
            cv_r2_scores.append(fold_r2)

        avg_mse_cv = np.mean(cv_mse_scores)
        avg_r2_cv = np.mean(cv_r2_scores)
        logging.info(f"CV Results - Fold-wise MSE and R2: {fold_results}")
        logging.info(f"CV Avg MSE: {avg_mse_cv}, CV Avg R2: {avg_r2_cv}")

        # Final training on full training data
        reg_final = CustomStructuredDataRegressor(
            max_trials=max_trials,
            overwrite=True,
            loss='mean_squared_error',
            project_name=f'{BASE_FOLDER}/1k-hb-all-tuning/final/{dataset_name}-final',
            tuner='greedy'
        )
        reg_final.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=4, callbacks=[nan_callback, oom_callback, thread_creation_error_callback, ram_usage_callback])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=950, restore_best_weights=True)
        tb = TensorBoard(log_dir=TB_FOLDER, histogram_freq=0, write_graph=True, write_images=True)
        ckpt = keras.callbacks.ModelCheckpoint(
            os.path.join(CKPT_FOLDER, 'loss-{val_loss:.4f}.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_freq='epoch'
        )

        # Predict and evaluate on training data
        y_train_pred = reg_final.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        logging.info(f"Training Data MSE: {mse_train}, Training Data R2: {r2_train}")

        # Predict and evaluate on test data
        y_test_pred = reg_final.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        current_r2 = r2_score(y_test, y_test_pred)
        logging.info(f"Test MSE: {mse_test}, Test R2: {current_r2}")

        # Predict and evaluate on validation data
        y_val_pred = reg_final.predict(X_validation)
        validation_results = pd.DataFrame({'Actual': y_validation, 'Predicted': y_val_pred.flatten()})
        validation_results.to_csv(validation_output_file, index=False)
        logging.info(f"Validation results saved to: {validation_output_file}")

        mse_val = mean_squared_error(y_validation, y_val_pred)
        r2_val = r2_score(y_validation, y_val_pred)

        # Save the model after each max_trials iteration
        model = reg_final.export_model()
        model.summary(print_fn=logging.info)

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        model.save(os.path.join(model_save_folder, f'{dataset}_best_model_{timestamp}'), save_format="tf")


        # Save hyperparameters
        previous_hyperparameters[hyperparameter_hash] = {
            'mse': mse_test,
            'r2': current_r2,
            'timestamp': timestamp
        }

        save_hyperparameters(previous_hyperparameters, hyperparameter_file)

        # Calculate test data error categories
        residual_test = y_test - y_test_pred.flatten()
        categories1_test = sum(abs(residual_test) <= 1)
        categories2_test = sum((abs(residual_test) > 1) & (abs(residual_test) <= 2))
        categories3_test = sum(abs(residual_test) > 2)

        # Calculate validation data error categories
        residual_validation = y_validation - y_val_pred.flatten()
        categories1_val = sum(abs(residual_validation) <= 1)
        categories2_val = sum((abs(residual_validation) > 1) & (abs(residual_validation) <= 2))
        categories3_val = sum(abs(residual_validation) > 2)

        # Update the result file writing to use the dataset-specific result_file path
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    ["Dataset", "CV-1 MSE", "CV-1 R2", "CV-2 MSE", "CV-2 R2", "CV-3 MSE", "CV-3 R2",
                     "CV-4 MSE", "CV-4 R2", "CV-5 MSE", "CV-5 R2", "CV Avg MSE", "CV Avg R2",
                     "MSE (Train)", "R2 (Train)",
                     "MSE (Test)", "R2 (Test)",
                     "MSE (Val)", "R2 (Val)",
                     "+-1 (Test)", "+-2 (Test)", "> +-2 (Test)", "+-1 (Val)", "+-2 (Val)", "> +-2 (Val)",
                     "Timestamp"]
                )
            writer.writerow(
                [dataset,
                 fold_results[0][1], fold_results[0][2],  # Fold 1 results
                 fold_results[1][1], fold_results[1][2],  # Fold 2 results
                 fold_results[2][1], fold_results[2][2],  # Fold 3 results
                 fold_results[3][1], fold_results[3][2],  # Fold 4 results
                 fold_results[4][1], fold_results[4][2],  # Fold 5 results
                 avg_mse_cv, avg_r2_cv,
                 mse_train, r2_train,
                 mse_test, current_r2,
                 mse_val, r2_val,
                 categories1_test, categories2_test, categories3_test,
                 categories1_val, categories2_val, categories3_val,
                 datetime.datetime.now()]
            )

        logging.info(f"Results saved to {result_file}")

    except Exception as e:
        logging.exception(f"An error occurred during training and prediction: {str(e)}")
        raise


# Main execution
# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='main_script.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Get all .parquet files in the DATASET_FOLDER
    datasets = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.parquet')]

    if not datasets:
        print(f"No .parquet files found in {DATASET_FOLDER}")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s) to process:")
    for dataset in datasets:
        print(f"- {dataset}")

    for dataset in datasets:
        logging.info(f"Starting training and prediction for dataset: {dataset}")
        try:
            train_and_predict(dataset)
            logging.info(f"Completed training and prediction for dataset: {dataset}")
        except Exception as e:
            logging.error(f"Error processing dataset {dataset}: {str(e)}")
            continue  # Move to the next dataset if there's an error

    logging.info("Finished processing all datasets.")
