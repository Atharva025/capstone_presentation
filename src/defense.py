import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
import json
import torchattacks

# We must re-import the classes and functions from our other modules
from src.baseline import BaselineIDS, PFCPDataset
from src.predictor import run_evaluation # CORRECTED IMPORT

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

# --- Model Wrapper for Attacks ---
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return torch.cat([-output, output], dim=1)


def train_robust_model(data_dir: str, model_dir: str, results_dir: str, epochs: int = 20, batch_size: int = 64, lr: float = 0.001):
    """
    Trains an adversarially robust IDS model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    train_data_path = os.path.join(data_dir, 'train.csv')
    val_data_path = os.path.join(data_dir, 'val.csv')
    columns_path = os.path.join(data_dir, 'feature_columns.json')

    train_dataset = PFCPDataset(train_data_path, columns_path=columns_path)
    val_dataset = PFCPDataset(val_data_path, columns_path=columns_path)
    
    # Use multiprocessing for data loading on CPU to prevent bottlenecks
    num_workers = os.cpu_count() if device == "cpu" else 0
    logging.info(f"Using {num_workers} worker processes for data loading.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model, Loss, and Optimizer ---
    input_features = len(train_dataset.columns)
    model = BaselineIDS(input_features=input_features).to(device)
    
    # Define the adversarial attack to train against (PGD is a strong choice)
    # The wrapped model is used here to ensure compatibility
    wrapped_model = ModelWrapper(model)
    atk = torchattacks.PGD(wrapped_model, eps=0.05, alpha=2/255, steps=7)

    # Weighted loss function to handle class imbalance
    # This calculation should be robust
    y_train_tensor = torch.tensor(train_dataset.y, dtype=torch.float32)
    pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum().clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_f1 = -1
    epochs_no_improve = 0
    patience = 5  # Stop after 5 epochs with no improvement

    logging.info("--- Starting Adversarial Training ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [T]")
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)

            # --- Adversarial Step ---
            # Generate adversarial data on the fly
            adv_data = atk(data, target.long())

            # --- Training Step ---
            optimizer.zero_grad()
            
            # Train on a mix of clean and adversarial data
            combined_data = torch.cat([data, adv_data], dim=0)
            combined_target = torch.cat([target, target], dim=0)
            
            outputs = model(combined_data)
            loss = criterion(outputs.squeeze(), combined_target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': running_loss / (train_pbar.n + 1)})

        # --- Validation Step ---
        # We need to save the validation data to a temporary file to use our evaluator
        temp_val_path = os.path.join(data_dir, "temp_val_for_eval.csv")
        pd.read_csv(val_data_path).to_csv(temp_val_path, index=False)
        
        # CORRECTED EVALUATION CALL
        val_metrics = run_evaluation(model_path=None, data_path=temp_val_path, columns_path=columns_path, model_in_memory=model)
        os.remove(temp_val_path) # Clean up temporary file

        if val_metrics:
            val_f1 = val_metrics['f1']
            logging.info(f"Epoch {epoch+1} - Validation F1-Score: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'robust_model.pt'))
                logging.info(f"Validation F1 improved to {best_f1:.4f}. Saving model.")
            else:
                epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f"No improvement in validation F1 for {patience} epochs. Early stopping.")
            break

    logging.info("--- Adversarial Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an adversarially robust IDS model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing processed data.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save evaluation results.")
    args = parser.parse_args()

    # We need to modify predictor to accept a model in memory
    # For now, let's just make sure the training runs
    # The evaluation logic in the training loop needs to be adjusted
    # to work without writing/reading the model from disk every time.
    
    # Let's adjust predictor.py to handle this case.
    # I will modify predictor.py and defense.py simultaneously.

    # This is a placeholder for the logic that will be moved to a refactored predictor
    def temp_evaluate_in_memory(model_to_eval, loader, device):
        model_to_eval.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                outputs = model_to_eval(data)
                preds = (torch.sigmoid(outputs).squeeze() > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
        
        from sklearn.metrics import f1_score
        return f1_score(all_labels, all_preds, zero_division=0)
    
    # --- This will be refactored into a cleaner structure later ---
    # For now, this is a direct fix to the user's problem

    # --- Data Loading (for this script) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_path = os.path.join(args.data_dir, 'train.csv')
    val_data_path = os.path.join(args.data_dir, 'val.csv')
    columns_path = os.path.join(args.data_dir, 'feature_columns.json')
    train_dataset = PFCPDataset(train_data_path, columns_path=columns_path)
    val_dataset = PFCPDataset(val_data_path, columns_path=columns_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    input_features = len(train_dataset.columns)
    model = BaselineIDS(input_features=input_features).to(device)
    wrapped_model = ModelWrapper(model)
    atk = torchattacks.PGD(wrapped_model, eps=0.05, alpha=2/255, steps=7)
    y_train_tensor = torch.tensor(train_dataset.y, dtype=torch.float32)
    pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum().clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_f1 = -1
    epochs_no_improve = 0
    patience = 5

    logging.info("--- Starting Adversarial Training ---")
    for epoch in range(20):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [T]")
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            adv_data = atk(data, target.long())
            combined_data = torch.cat([data, adv_data], dim=0)
            combined_target = torch.cat([target, target], dim=0)
            optimizer.zero_grad()
            outputs = model(combined_data)
            loss = criterion(outputs.squeeze(), combined_target)
            loss.backward()
            optimizer.step()
        
        val_f1 = temp_evaluate_in_memory(model, val_loader, device)
        logging.info(f"Epoch {epoch+1} - Validation F1-Score: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'robust_model.pt'))
            logging.info(f"Validation F1 improved to {best_f1:.4f}. Saving model.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f"No improvement in validation F1 for {patience} epochs. Early stopping.")
            break
    logging.info("--- Adversarial Training Complete ---")

