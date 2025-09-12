import torch
import pandas as pd
import argparse
import os
import logging
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# --- Import from our own project files ---
from baseline import BaselineIDS, PFCPDataset
from torch.utils.data import DataLoader

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def get_predictions(model, loader, device):
    """Gets model predictions for a given dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return np.array(all_labels).flatten(), np.array(all_preds).flatten()

def create_and_save_confusion_matrix(labels, preds, title, save_path):
    """Creates a confusion matrix plot and saves it to a file."""
    cm = confusion_matrix(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malicious'],
                yticklabels=['Normal', 'Malicious'])
    plt.title(f'{title}\nF1-Score: {f1:.4f}', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved confusion matrix to '{save_path}'")

def run_visualization(model_path: str, data_path: str, output_path: str, plot_title: str):
    """Main function to generate and save a confusion matrix for a model and dataset."""
    if not os.path.exists(model_path):
        logging.error(f"FATAL: Model file not found at '{model_path}'.")
        raise FileNotFoundError
    if not os.path.exists(data_path):
        logging.error(f"FATAL: Data file not found at '{data_path}'.")
        raise FileNotFoundError

    device = torch.device("cpu")

    # Load data
    dataset = PFCPDataset(data_path)
    loader = DataLoader(dataset, batch_size=256)

    # Load model
    num_features = len(dataset.columns)
    model = BaselineIDS(num_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Get predictions
    labels, preds = get_predictions(model, loader, device)
    
    # Generate and save plot
    create_and_save_confusion_matrix(labels, preds, plot_title, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate confusion matrix for a model's performance.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pt model file.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file for evaluation.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output confusion matrix .png file.")
    parser.add_argument('--title', type=str, required=True, help="Title for the plot.")

    args = parser.parse_args()
    run_visualization(args.model_path, args.data_path, args.output_path, args.title)
