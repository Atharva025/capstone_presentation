import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
import torchattacks

# We must re-import these classes to use them for model loading and data handling
from src.baseline import BaselineIDS, PFCPDataset

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

# --- DEFINITIVE FIX: The Model Wrapper ---
# This class acts as an adapter between our single-output binary model
# and the multi-class format expected by the torchattacks library.
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Get the single logit output from our model (e.g., shape [batch_size, 1])
        output = self.model(x)
        # Transform it into a 2-class output format [batch_size, 2]
        # [logit] -> [-logit, logit]
        # This makes a positive logit correspond to class 1, and negative to class 0.
        return torch.cat([-output, output], dim=1)
# --- END FIX ---


def generate_attacks(model: nn.Module, loader: torch.utils.data.DataLoader, device: str, attack_type: str = 'fgsm', epsilon: float = 0.05):
    """
    Generates adversarial examples for a given model and data loader.
    """
    logging.info(f"Generating adversarial examples using {attack_type.upper()} attack...")
    model.eval()

    if attack_type == 'fgsm':
        atk = torchattacks.FGSM(model, eps=epsilon)
    elif attack_type == 'pgd':
        atk = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
    else:
        raise ValueError("Unsupported attack type. Choose 'fgsm' or 'pgd'.")

    adv_samples = []
    original_labels = []

    for data, target in tqdm(loader, desc=f"Attacking with {attack_type.upper()}"):
        data, target = data.to(device), target.to(device)
        
        # The attack library's internal loss function demands the target tensor be of type long.
        target = target.long()

        adv_data = atk(data, target)
        adv_samples.append(adv_data.cpu().detach().numpy())
        original_labels.append(target.cpu().detach().numpy())

    adv_x = np.concatenate(adv_samples)
    adv_y = np.concatenate(original_labels)

    # Create DataFrame
    adv_df = pd.DataFrame(adv_x, columns=loader.dataset.columns)
    adv_df['Label'] = adv_y.astype(int)

    return adv_df


def run_attack_generation(data_dir: str, model_path: str, adversarial_dir: str, attack: str, epsilon: float):
    """
    Main function to orchestrate the attack generation process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load the test dataset (which is already scaled)
    test_data_path = os.path.join(data_dir, 'test.csv')
    columns_path = os.path.join(data_dir, 'feature_columns.json')

    test_dataset = PFCPDataset(test_data_path, columns_path=columns_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the trained baseline model
    input_features = len(test_dataset.columns)
    base_model = BaselineIDS(input_features=input_features).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Successfully loaded model from '{model_path}'")

    # --- DEFINITIVE FIX: Use the wrapper ---
    # We pass our loaded model into the wrapper before giving it to the attack generator.
    wrapped_model = ModelWrapper(base_model)
    # --- END FIX ---

    # Generate attacks using the wrapped model
    adversarial_df = generate_attacks(wrapped_model, test_loader, device, attack_type=attack, epsilon=epsilon)

    # Save the adversarial dataset
    os.makedirs(adversarial_dir, exist_ok=True)
    output_filename = f"{attack}_adversaries_eps_{epsilon}.csv"
    output_path = os.path.join(adversarial_dir, output_filename)

    adversarial_df.to_csv(output_path, index=False)
    logging.info(f"Saved adversarial dataset to '{output_path}'")
    logging.info("--- ATTACK GENERATION COMPLETE ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate adversarial attacks on the baseline IDS.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing processed data.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained baseline model.")
    parser.add_argument('--adversarial_dir', type=str, required=True, help="Directory to save adversarial samples.")
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd'], help="Type of attack to generate.")
    parser.add_argument('--epsilon', type=float, default=0.05, help="Epsilon value for FGSM/PGD attacks.")

    args = parser.parse_args()
    run_attack_generation(args.data_dir, args.model_path, args.adversarial_dir, args.attack, args.epsilon)

