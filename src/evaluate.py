import torch
import pandas as pd
import argparse
import os
import logging
import json

# --- Import from our own project files ---
from baseline import BaselineIDS, PFCPDataset, evaluate
from torch.utils.data import DataLoader

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def run_evaluation(model_path: str, data_path: str, results_path: str):
    """
    Loads a trained model and evaluates it on a specified dataset.

    Args:
        model_path (str): Path to the trained .pt model file.
        data_path (str): Path to the .csv data file to evaluate on.
        results_path (str): Path to save the resulting metrics JSON file.
    """
    if not os.path.exists(model_path):
        logging.error(f"FATAL: Model file not found at '{model_path}'.")
        raise FileNotFoundError
    if not os.path.exists(data_path):
        logging.error(f"FATAL: Data file not found at '{data_path}'.")
        raise FileNotFoundError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device.type.upper()}")

    # Load the dataset for evaluation
    logging.info(f"Loading evaluation data from '{data_path}'...")
    eval_dataset = PFCPDataset(data_path)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Initialize model architecture and load the trained weights
    num_features = len(eval_dataset.columns)
    model = BaselineIDS(num_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Successfully loaded model from '{model_path}'")

    # Run evaluation
    logging.info("Starting evaluation...")
    metrics, _ = evaluate(model, eval_loader, device)

    # --- Log results and save ---
    logging.info("--- EVALUATION RESULTS ---")
    for key, value in metrics.items():
        logging.info(f"{key.capitalize()}: {value:.4f}")

    # Ensure the directory for the results exists
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Evaluation metrics saved to '{results_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained IDS model on a given dataset.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained baseline_model.pt file.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file for evaluation (e.g., adversarial data).")
    parser.add_argument('--results_path', type=str, required=True, help="Path to save the output metrics JSON file.")

    args = parser.parse_args()
    run_evaluation(args.model_path, args.data_path, args.results_path)

    # Example Usage from your terminal:
    # python src/evaluate.py --model_path models/baseline_model.pt --data_path data/adversarial/fgsm_adversaries_eps_0.05.csv --results_path results/baseline/robustness_report_fgsm_0.05.json
