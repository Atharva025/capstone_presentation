import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# We must re-import the model's class definition to load the saved model
from src.baseline import BaselineIDS

def run_evaluation(model_path: str, data_path: str, columns_path: str):
    """
    Evaluates a trained model on a given dataset.
    This function loads all necessary assets from the specified file paths,
    making it a robust and self-contained evaluation utility.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(columns_path, 'r') as f:
            feature_columns = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Columns file not found at {columns_path}")
        return None

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {data_path}")
        return None

    if 'Label' not in df.columns:
         print(f"FATAL ERROR: The data file at {data_path} is missing the required 'Label' column for evaluation.")
         return None

    if not all(col in df.columns for col in feature_columns):
        print("FATAL ERROR: Data file is missing required feature columns.")
        return None

    features = df[feature_columns]
    labels = df['Label'].values
    features_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)

    # --- Model Loading Logic ---
    model = BaselineIDS(input_features=len(feature_columns)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {model_path}")
        return None
    
    # --- Get Predictions ---
    model.eval()
    all_preds = []
    with torch.no_grad():
        # Process in batches to avoid memory errors on large test sets
        for i in range(0, len(features_tensor), 64):
            batch = features_tensor[i:i+64]
            outputs = model(batch)
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int().cpu().numpy()
            if preds.ndim == 0: # Handle single-item batch case
                preds = [preds.item()]
            all_preds.extend(preds)
            
    # --- Calculate Metrics ---
    metrics = {
        "f1": f1_score(labels, all_preds, zero_division=0),
        "precision": precision_score(labels, all_preds, zero_division=0),
        "recall": recall_score(labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(labels, all_preds)
    }

    return metrics

