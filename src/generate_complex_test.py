import pandas as pd
import argparse
import os
import logging
from sklearn.model_selection import train_test_split

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def create_complex_test_file(raw_data_path: str, output_path: str, n_samples: int = 200):
    """
    Generates a complex, statistically representative test file by performing
    stratified sampling on the raw dataset.

    Args:
        raw_data_path (str): Path to the full, raw CSV dataset.
        output_path (str): Path to save the generated complex test CSV file.
        n_samples (int): The total number of samples for the output file.
    """
    logging.info(f"Loading raw data from '{raw_data_path}' to generate complex test file...")

    if not os.path.exists(raw_data_path):
        logging.error(f"FATAL: Raw data file not found at '{raw_data_path}'.")
        raise FileNotFoundError(f"Raw data file not found at '{raw_data_path}'.")

    try:
        df = pd.read_csv(raw_data_path, low_memory=False)
        # Ensure the label column has no leading/trailing whitespace
        df['Label'] = df['Label'].str.strip()
    except Exception as e:
        logging.error(f"Failed to read raw CSV: {e}")
        raise

    if len(df) < n_samples:
        logging.error(f"FATAL: Raw dataset has fewer rows ({len(df)}) than requested sample size ({n_samples}).")
        raise ValueError("Cannot sample more rows than exist in the dataset.")

    # We use train_test_split as a powerful tool for stratified sampling.
    # We are not actually "splitting" but rather "sampling off" a chunk of the data.
    # The 'y' for stratification is the 'Label' column.
    _, complex_sample_df = train_test_split(
        df,
        test_size=n_samples,
        random_state=42,
        stratify=df['Label']
    )

    # Save the complex sample to CSV
    complex_sample_df.to_csv(output_path, index=False)
    logging.info(f"Successfully created complex test file with {len(complex_sample_df)} rows at '{output_path}'")
    logging.info(f"Class distribution of the new test file:\n{complex_sample_df['Label'].value_counts(normalize=True)}")
    logging.info(f"Original dataset class distribution for comparison:\n{df['Label'].value_counts(normalize=True)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a complex, stratified sample file from the raw dataset.")
    parser.add_argument('--raw_data_path', type=str, required=True, help="Path to the raw CSV dataset.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output test CSV file.")

    args = parser.parse_args()
    create_complex_test_file(args.raw_data_path, args.output_path)
