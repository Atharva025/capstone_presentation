import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse
import os
import logging
import json
from tqdm import tqdm
import multiprocessing

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

# --- Dataset Class ---
class PFCPDataset(Dataset):
    """
    Custom PyTorch Dataset for the pre-scaled PFCP IDS data.
    """
    def __init__(self, data_path: str, columns_path: str):
        """
        Args:
            data_path (str): Path to the processed (scaled) CSV file.
            columns_path (str): Path to the JSON file containing feature column names.
        """
        df = pd.read_csv(data_path)

        # Load the feature columns that the model was trained on
        with open(columns_path, 'r') as f:
            self.columns = json.load(f)

        # Ensure the dataframe only contains the expected feature columns + Label
        self.y = torch.tensor(df['Label'].values, dtype=torch.float32)
        
        # Reorder/select columns to match the training order precisely
        features_df = df[self.columns]
        self.X = torch.tensor(features_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Model Architecture ---
class BaselineIDS(nn.Module):
    """
    The neural network architecture for our baseline IDS.
    Simple, but robustly defined with BatchNorm and Dropout.
    """
    def __init__(self, input_features: int):
        super(BaselineIDS, self).__init__()
        self.net = nn.Sequential(
            # --- HIDDEN LAYER 1 ---
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            # --- HIDDEN LAYER 2 ---
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # --- HIDDEN LAYER 3 ---
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # --- OUTPUT LAYER ---
            nn.Linear(64, 1) # Single output for binary classification
        )

    def forward(self, x):
        return self.net(x)

# --- Training and Evaluation Logic ---
def train_and_evaluate(data_dir: str, model_dir: str, results_dir: str):
    """
    Main function to run the training and evaluation pipeline.
    """
    # --- Hyperparameters and Setup ---
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 5 # For early stopping

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    columns_path = os.path.join(data_dir, 'feature_columns.json')

    train_dataset = PFCPDataset(train_path, columns_path)
    val_dataset = PFCPDataset(val_path, columns_path)
    test_dataset = PFCPDataset(test_path, columns_path)

    # Optimized for CPU training
    num_workers = multiprocessing.cpu_count()
    logging.info(f"Using {num_workers} processes for data loading.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # --- Class Weighting for Imbalanced Data ---
    # This is critical. It punishes the model more for misclassifying the minority class.
    train_labels = train_dataset.y.numpy()
    class_counts = np.bincount(train_labels.astype(int))
    weight_for_0 = 1.0 / class_counts[0]
    weight_for_1 = 1.0 / class_counts[1]
    class_weight = torch.tensor([weight_for_1 / (weight_for_0 + weight_for_1)], dtype=torch.float32).to(device)
    
    logging.info(f"Class counts (0/1): {class_counts}")
    logging.info(f"Calculated class weight for malicious class (1): {class_weight.item():.4f}")

    # --- Model, Loss, and Optimizer ---
    input_features = len(train_dataset.columns)
    model = BaselineIDS(input_features=input_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    best_val_f1 = -1
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Loop ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).squeeze().round()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        logging.info(
            f"Epoch {epoch+1} | Val F1: {val_f1:.4f} | "
            f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
        )

        # --- Early Stopping and Model Checkpointing ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "baseline_model.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Validation F1 improved. Saving model to '{model_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logging.info(f"No improvement for {PATIENCE} epochs. Stopping early.")
                break
    
    # --- Final Evaluation on Test Set ---
    logging.info("--- Training complete. Evaluating on the test set with the best model. ---")
    model.load_state_dict(torch.load(model_path)) # Load the best model
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[Test Evaluation]")
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).squeeze().round()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    logging.info(
        f"Test Set Results: F1={test_f1:.4f}, Precision={test_precision:.4f}, "
        f"Recall={test_recall:.4f}, Accuracy={test_accuracy:.4f}"
    )
    
    # --- Save Metrics Report ---
    metrics = {
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "accuracy": test_accuracy
    }
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "metrics_report.json")
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved final metrics report to '{report_path}'")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline IDS model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing processed train/val/test CSVs.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save the results report.")
    
    args = parser.parse_args()
    train_and_evaluate(args.data_dir, args.model_dir, args.results_dir)

