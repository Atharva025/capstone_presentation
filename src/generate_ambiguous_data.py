import pandas as pd
import argparse
import os

def generate_ambiguous_file(raw_data_path: str, output_path: str):
    """
    Creates a test CSV file with deliberately ambiguous column names
    to test the dashboard's manual mapping feature.

    It takes a sample from the raw data and renames a few key columns
    to common, non-exact variations.
    """
    print("--- Generating Ambiguous Test Dataset ---")
    if not os.path.exists(raw_data_path):
        print(f"FATAL: Raw data file not found at '{raw_data_path}'.")
        raise FileNotFoundError

    try:
        df = pd.read_csv(raw_data_path, nrows=20) # Take a small sample
        
        # Define the remapping for ambiguity
        rename_map = {
            'Flow Duration': 'duration',
            'Tot Fwd Pkts': 'fwd_packets',
            'Tot Bwd Pkts': 'bwd_packets',
            'Fwd Pkt Len Max': 'max_fwd_len',
            'Flow IAT Mean': 'flow_inter_arrival_avg'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ Successfully created ambiguous test file at '{output_path}'")
        print("A few columns have been renamed (e.g., 'Flow Duration' is now 'duration').")
        print("Upload this file to the dashboard to test the manual mapping UI.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a CSV with ambiguous column names for testing the dashboard's mapping feature."
    )
    parser.add_argument(
        '--raw_data_path',
        type=str,
        required=True,
        help="Path to the original raw CSV dataset."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path to save the new ambiguous CSV file."
    )

    args = parser.parse_args()
    generate_ambiguous_file(args.raw_data_path, args.output_path)
