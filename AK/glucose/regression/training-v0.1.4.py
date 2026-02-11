import autokeras as ak
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math
import matplotlib.pyplot as plt


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


def step4_train_model():
        """Step 4: Train TensorFlow model."""
        print("\n🤖 Step 4: Model Training")
        print("=" * 40)

        fixed_parquet_dir = "/home/apc-3/PycharmProjects/PythonProjectAK/1k-gl-train/dataset/dataset-1k-glucose/pls/dataset-1k-glucose-pls-top10-parquet"

        # Train ONLY on this specific parquet file
        target_parquet = "Lablink_1k_glucose_SNV_top_10.parquet"
        data_file = os.path.join(fixed_parquet_dir, target_parquet)

        if not os.path.exists(data_file):
            print(f"❌ Target parquet not found: {data_file}")
            return False

        parquet_name = target_parquet
        print(f"Training on: {parquet_name}")
        
        # Load data
        data = pd.read_parquet(data_file)
        X = data.iloc[:, 2:]  # Spectral data
        target_name = "glucose"
        y = data[target_name]  # Target variable
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize the CustomStructuredDataRegressor
        reg = CustomStructuredDataRegressor(
            max_trials=100,  # Max number of models to test
            overwrite=True,
            loss='mean_squared_error',
            project_name='spectral_tuning',
            tuner='greedy'
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
        ]

        # Train the model
        # AutoKeras will use the validation_data for monitoring and early stopping.
        reg.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            callbacks=callbacks
        )

        # Export the best model found by the tuner
        model = reg.export_model()
        model.summary()

        saved_dir = '/home/apc-3/PycharmProjects/PythonProjectAK/1k-gl-train/model'
        os.makedirs(saved_dir, exist_ok=True)

        # Use the parquet base name as model name
        model_name_base = os.path.splitext(parquet_name)[0]

        saved_subdir = os.path.join(saved_dir, f'{model_name_base}_{reg.max_trials}')
        os.makedirs(saved_subdir, exist_ok=True)

        # Save in TensorFlow SavedModel format (similar to AutoKeras approach)
        saved_model_path = os.path.join(saved_subdir, 'model-tensorflow')
        model.save(saved_model_path, save_format="tf")
        
        # Optionally keep Keras format as backup
        keras_model_path = os.path.join(saved_subdir, f'{model_name_base}.keras')
        model.save(keras_model_path, save_format="keras")

        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"✓ Model trained - Test MAE: {test_mae:.4f}")
        print(f"✓ Model saved to: {saved_model_path}")

        # ---------- 1) Metrics score ----------
        # Use explicit predictions for R2 / RMSE / RME
        y_pred = model.predict(X_test).flatten()
        y_true = y_test.values.flatten()

        # R2 & RMSE
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        print(f"R²:   {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # === Save metrics into a single CSV in output dir ===
        output_dir = "/home/apc-3/PycharmProjects/PythonProjectAK/1k-gl-train/output"
        os.makedirs(output_dir, exist_ok=True)

        output_subdir = os.path.join(output_dir, model_name_base)
        os.makedirs(output_subdir, exist_ok=True)

        metrics_csv = os.path.join(output_dir, "ALL_training_metrics.csv")

        metrics_row = {
            "parquet_file": parquet_name,
            "max_trials": reg.max_trials,
            "r2": r2,
            "rmse": rmse,
            "mae": test_mae
        }

        # Append or create CSV
        if os.path.exists(metrics_csv):
            df_old = pd.read_csv(metrics_csv)
            df_new = pd.concat([df_old, pd.DataFrame([metrics_row])], ignore_index=True)
        else:
            df_new = pd.DataFrame([metrics_row])

        df_new.to_csv(metrics_csv, index=False)
        print(f"✓ Metrics appended to: {metrics_csv}")

        # ---------- 2) Save actual vs predicted to CSV ----------
        pred_csv_path = os.path.join(output_subdir, f"predictions_{model_name_base}_{reg.max_trials}.csv" )

        pred_df = pd.DataFrame({
            "actual_glucose": y_true,
            "predicted_glucose": y_pred
        })
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"✓ Predictions saved to: {pred_csv_path}")

        # ---------- 3) Plot actual vs predicted ----------
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.7)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("Actual Glucose")
        plt.ylabel("Predicted Glucose")
        plt.title(f"Actual vs Predicted Glucose - {model_name_base}")
        plt.grid(True)

        plot_path = os.path.join(output_subdir, f"predictions_plot_{model_name_base}_{reg.max_trials}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Plot saved to: {plot_path}")

        return saved_model_path

if __name__ == "__main__":
    step4_train_model()
