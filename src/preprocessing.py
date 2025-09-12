import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os
import logging
import warnings
import json

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

# --- Suppress Harmless Warnings ---
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)


def preprocess_data(raw_data_path: str, processed_data_dir: str, scaler_path_to_use: str = None, columns_path_to_use: str = None, save_path: str = None):
    """
    Loads, cleans, preprocesses, and splits/saves the dataset.
    
    This function has two modes:
    1. Training Mode (default): Splits data into train/val/test, fits a new scaler,
       and saves all artifacts.
    2. Inference Mode (when `save_path` is provided): Preprocesses a single file for prediction,
       using a pre-fitted scaler and column list.
    """
    if not os.path.exists(raw_data_path):
        logging.error(f"FATAL: Raw data file not found at '{raw_data_path}'.")
        return False

    logging.info(f"Starting preprocessing for: {raw_data_path}")

    try:
        df = pd.read_csv(raw_data_path, low_memory=False)
        logging.info(f"Successfully loaded raw data. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"FATAL: Failed to load CSV file. Error: {e}")
        return False

    # --- Data Cleaning & Sanitization ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.isnull().values.any():
        # Use median for imputation as it's more robust to outliers
        df.fillna(df.median(numeric_only=True), inplace=True)
    
    # --- Target Encoding ---
    if 'Label' in df.columns:
        try:
            df['Label'] = df['Label'].apply(lambda x: 1 if isinstance(x, str) and x.strip().lower() == 'malicious' else 0)
        except Exception:
            logging.error("Could not process 'Label' column. It may contain non-string values.")
            return False
    else:
        logging.error("FATAL: 'Label' column not found in the dataset.")
        return False

    # --- Feature and Target Separation ---
    y = df['Label']
    
    # In training mode, we define the feature set. In inference, we use the pre-defined set.
    if columns_path_to_use and os.path.exists(columns_path_to_use):
        with open(columns_path_to_use, 'r') as f:
            numeric_features = json.load(f)
        
        # Ensure all required columns are present
        if not all(col in df.columns for col in numeric_features):
            logging.error("The provided data is missing columns that the model was trained on.")
            return False
        X = df[numeric_features]
    else:
        # First-time run: identify numeric features, excluding obvious non-features
        X = df.drop(columns=['Label', 'Data_Source'], errors='ignore')
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        # Drop any remaining non-numeric columns that slipped through
        X = X[numeric_features]
        logging.info(f"Identified {len(numeric_features)} numeric features for modeling.")

    # --- Scaling and Splitting Logic ---
    
    # INFERENCE MODE: Process a single file for prediction
    if save_path:
        if not scaler_path_to_use or not os.path.exists(scaler_path_to_use):
            logging.error("FATAL: In inference mode, a valid scaler path must be provided.")
            return False
        scaler = joblib.load(scaler_path_to_use)
        X_scaled = scaler.transform(X)
        processed_df = pd.DataFrame(X_scaled, columns=numeric_features)
        processed_df['Label'] = y.values
        processed_df.to_csv(save_path, index=False)
        logging.info(f"Saved processed inference file to '{save_path}'")
        return True

    # TRAINING MODE: Split data and fit a new scaler
    else:
        logging.info("Performing stratified split: 70% Train, 15% Validation, 15% Test")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val, shuffle=True
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Create DataFrames from scaled data
        train_df = pd.DataFrame(X_train_scaled, columns=numeric_features)
        train_df['Label'] = y_train.values
        val_df = pd.DataFrame(X_val_scaled, columns=numeric_features)
        val_df['Label'] = y_val.values
        test_df = pd.DataFrame(X_test_scaled, columns=numeric_features)
        test_df['Label'] = y_test.values

        os.makedirs(processed_data_dir, exist_ok=True)
        train_df.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(processed_data_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)

        # Save the scaler and feature columns for future use
        scaler_path_to_save = os.path.join(processed_data_dir, 'scaler.joblib')
        columns_path_to_save = os.path.join(processed_data_dir, 'feature_columns.json')
        joblib.dump(scaler, scaler_path_to_save)
        with open(columns_path_to_save, 'w') as f:
            json.dump(numeric_features, f)
            
        logging.info(f"Saved artifacts: 'train.csv', 'val.csv', 'test.csv', 'scaler.joblib', 'feature_columns.json'")
        logging.info("--- PREPROCESSING COMPLETE ---")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess the 5G PFCP-Layer Intrusion Dataset.")
    parser.add_argument('--raw_data_path', type=str, required=True, help="Path to the raw CSV dataset.")
    parser.add_argument('--processed_data_dir', type=str, required=True, help="Directory to save the processed data files.")
    args = parser.parse_args()
    preprocess_data(args.raw_data_path, args.processed_data_dir)

