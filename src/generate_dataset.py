import pandas as pd
import argparse
import os
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def create_demo_file(raw_data_path: str, output_path: str, n_samples: int = 5):
    """
    Intelligently samples from the raw dataset to create a small, balanced
    demonstration file for the CLI tool.

    Args:
        raw_data_path (str): Path to the full, raw CSV dataset.
        output_path (str): Path to save the generated demo CSV file.
        n_samples (int): The number of samples to take from each class.
    """
    logging.info(f"Loading raw data from '{raw_data_path}' to generate demo file...")
    
    if not os.path.exists(raw_data_path):
        logging.error(f"FATAL: Raw data file not found at '{raw_data_path}'. Cannot generate demo file.")
        raise FileNotFoundError(f"Raw data file not found at '{raw_data_path}'.")

    try:
        df = pd.read_csv(raw_data_path, low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read raw CSV: {e}")
        raise

    # Separate the classes
    malicious_df = df[df['Label'].str.strip().str.lower() == 'malicious']
    normal_df = df[df['Label'].str.strip().str.lower() == 'normal']

    if len(malicious_df) < n_samples or len(normal_df) < n_samples:
        logging.error("FATAL: Not enough samples in the raw dataset to generate a balanced demo file.")
        raise ValueError("Raw dataset does not contain enough samples of each class.")

    # Sample from each class
    malicious_sample = malicious_df.sample(n=n_samples, random_state=42)
    normal_sample = normal_df.sample(n=n_samples, random_state=42)

    # Combine and shuffle
    demo_df = pd.concat([malicious_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    demo_df.to_csv(output_path, index=False)
    logging.info(f"Successfully created demo file with {len(demo_df)} rows at '{output_path}'")
    logging.info(f"Class distribution:\n{demo_df['Label'].value_counts()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a small, balanced demo file from the raw dataset.")
    parser.add_argument('--raw_data_path', type=str, required=True, help="Path to the raw CSV dataset.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output demo CSV file.")

    args = parser.parse_args()
    create_demo_file(args.raw_data_path, args.output_path)

