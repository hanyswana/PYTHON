import os
import logging
import datetime
import csv
import pandas as pd
import numpy as np
import prefect
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from autokeras import StructuredDataRegressor
from prefect import flow, task, exceptions
from prefect.tasks import task_input_hash
from datetime import timedelta
from keras.callbacks import EarlyStopping, TensorBoard
import keras.callbacks
import json
import hashlib
import autokeras as ak
from autokeras import Node, Input, StructuredDataInput
from autokeras.engine import block as auto_model
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

# Configure logging
logging.basicConfig(level=logging.INFO, filename='experiment_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Base folder location
BASE_FOLDER = '/home/applied-photonics-core/PycharmProjects/autokeras-2/esp32-model-dev'

# File locations
DATASET_FOLDER = os.path.join(BASE_FOLDER, 'dataset/test-automate-dataset')
TB_FOLDER = os.path.join(BASE_FOLDER, 'lablink-108-SAFE (copy)/batch-size-16-4070ti-prefect/tb-topLV-batch-size')
CKPT_FOLDER = os.path.join(BASE_FOLDER, 'checkpoint')
MODEL_SAVE_FOLDER = os.path.join(BASE_FOLDER, 'all-best-model-L1L2-dropout')
HYPERPARAMETER_FOLDER = os.path.join(BASE_FOLDER, 'all-model-hyperparameter-L1L2-dropout')
RESULT_FOLDER = os.path.join(BASE_FOLDER, 'all-model-result-L1L2-dropout')
VALIDATION_FOLDER = os.path.join(BASE_FOLDER, 'dataset/lablink_test_data/output')



def get_validation_file(dataset_name):
    # Remove the '.parquet' extension
    dataset_name = dataset_name.replace('.parquet', '')

    # Split the dataset name
    parts = dataset_name.split('_')

    # The preprocessing steps are the parts after '128hb'
    preprocessing_steps = '_'.join(parts[parts.index('128hb') + 1:])

    validation_file_name = f"REVA LABLINK 2024_22_{preprocessing_steps}.csv"
    validation_file_path = os.path.join(VALIDATION_FOLDER, validation_file_name)

    if os.path.exists(validation_file_path):
        return validation_file_path
    else:
        logging.warning(f"Validation file not found: {validation_file_path}")
        return None


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
            num_layers=hp.Int('num_layers', min_value=1, max_value=5),
            num_units=hp.Int('units', min_value=4, max_value=64, step=6),
            dropout=hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.05),
            use_batchnorm=False,
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1', min_value=1e-6, max_value=1e-3, sampling='log'),
                l2=hp.Float('l2', min_value=1e-6, max_value=1e-3, sampling='log')
            )
        )(output_node)

        output_node = ak.RegressionHead()(output_node)

        return ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True)


#################################


@flow(retries=5, retry_delay_seconds=1800)
def train_and_predict(dataset):
    try:
        file = os.path.join(DATASET_FOLDER, dataset)
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]

        # Dataset-specific paths
        model_save_folder = os.path.join(MODEL_SAVE_FOLDER, dataset_name)
        result_file = os.path.join(RESULT_FOLDER, f'results-{dataset_name}.csv')
        validation_file = get_validation_file(dataset)

        os.makedirs(model_save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        if not validation_file:
            logging.error(f"Validation file not found for dataset {dataset}. Skipping this dataset.")
            return

        df = pd.read_parquet(file)
        logging.info(f"Loaded file: {file}")

        # Get the column names from the training data
        train_columns = df.columns.tolist()
        feature_columns = [col for col in train_columns if col not in ['Hb 67', 'Sample', 'Hb 97']]

        X = df[feature_columns]
        y = df['Hb 97']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Load validation data
        validation_data = pd.read_csv(validation_file)

        # Ensure validation data has the same columns as training data
        missing_columns = set(feature_columns) - set(validation_data.columns)
        if missing_columns:
            logging.warning(f"Missing columns in validation data: {missing_columns}")
            raise ValueError("Validation data does not have all required columns")

        x_validation = validation_data[feature_columns]
        y_validation = validation_data['Hb 97']

        logging.info(f"Validation data shape: {x_validation.shape}")

        max_trials = 5
        increment_trials = 10
        desired_r2 = 0.1
        current_r2 = 0.0

        # # Extract dataset name from the file
        # dataset_name = os.path.splitext(file)[0]

        # Dynamically create the hyperparameter file path
        hyperparameter_file = os.path.join(HYPERPARAMETER_FOLDER, f'{dataset_name}_hyperparameter.json')

        os.makedirs(model_save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        os.makedirs(os.path.dirname(hyperparameter_file), exist_ok=True)

        previous_hyperparameters = load_hyperparameters(hyperparameter_file)

        while current_r2 < desired_r2:
            hyperparameters = {
                'max_trials': max_trials,
                'project_name': f'{dataset_name}-tuning',
                'output_dim': 1,
                'tuner': 'greedy'
            }

            while True:
                hyperparameter_hash = get_hyperparameter_hash(hyperparameters)
                if hyperparameter_hash in previous_hyperparameters:
                    logging.info(f"Skipping previously used hyperparameters: {hyperparameters}")
                    max_trials += increment_trials
                    hyperparameters['max_trials'] = max_trials  # Update the hyperparameters dictionary
                    logging.info(f"Updated hyperparameters: {hyperparameters}")
                else:
                    break

            # Proceed with the current hyperparameter setting
            reg = CustomStructuredDataRegressor(
                max_trials=max_trials,
                overwrite=True,
                loss='mean_squared_error',
                project_name=f'{dataset_name}-tuning',
                tuner='greedy'
            )

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

            nan_callback = NaNCallback()
            oom_callback = OOMCallback()
            thread_creation_error_callback = ThreadCreationErrorCallback()
            ram_usage_callback = RAMUsageCallback(threshold=98)

            reg.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=4,
                callbacks=[nan_callback, oom_callback, thread_creation_error_callback, ram_usage_callback]
            )


            # Predict on test data
            predicted_y = reg.predict(X_test)
            # Evaluate on test data
            mse = reg.evaluate(X_test, y_test)
            current_r2 = r2_score(y_test, predicted_y)
            logging.info(f"MSE: {mse}, R2: {current_r2}")

            # Predict on validation data
            predicted_y_test = reg.predict(x_validation)
            # Evaluate on validation data
            mse_test = reg.evaluate(x_validation, y_validation)
            r2_test = r2_score(y_validation, predicted_y_test)

            # Save the model after each max_trials iteration
            model = reg.export_model()
            model.summary(print_fn=logging.info)

            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            model.save(os.path.join(model_save_folder, f'{dataset}_best_model_{timestamp}'), save_format="tf")

            # Save hyperparameters
            previous_hyperparameters[hyperparameter_hash] = {
                'mse': mse,
                'r2': current_r2,
                'timestamp': timestamp
            }

            save_hyperparameters(previous_hyperparameters, hyperparameter_file)

            if current_r2 >= desired_r2:
                break

            max_trials += increment_trials

            # calculate test data error category
            residual = y_test - predicted_y.flatten()
            categories1 = sum(abs(residual) <= 1)
            categories2 = sum((abs(residual) > 1) & (abs(residual) <= 2))
            categories3 = sum(abs(residual) > 2)

            # calculate evaluate data error category
            residual_evaluate = y_validation - predicted_y_test.flatten()
            categories1_eval = sum(abs(residual_evaluate) <= 1)
            categories2_eval = sum((abs(residual_evaluate) > 1) & (abs(residual_evaluate) <= 2))
            categories3_eval = sum(abs(residual_evaluate) > 2)

            # Update the result file writing to use the dataset-specific result_file path
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(
                        ["CSV", "MSE", "R2", "+-1", "+-2", "> +-2", "Timestamp", "val_MSE", "val_R2", "val_+-1",
                         "val_+-2",
                         "> val_+-2"])
                writer.writerow(
                    [dataset, mse[0], current_r2, categories1, categories2, categories3, timestamp, mse_test[0],
                     r2_test,
                     categories1_eval, categories2_eval, categories3_eval])

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
