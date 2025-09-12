import click
import json
import os
import tempfile
import shutil

# We must be able to import the core logic from our other scripts
from src.predictor import run_evaluation
from src.preprocessing import preprocess_data

# --- Define constants for model and scaler paths ---
# This makes our CLI robust and easy to update.
MODEL_DIR = 'models'
DATA_DIR = 'data'
ROBUST_MODEL_PATH = os.path.join(MODEL_DIR, 'robust_model.pt')
SCALER_PATH = os.path.join(DATA_DIR, 'processed', 'scaler.joblib')
COLUMNS_PATH = os.path.join(DATA_DIR, 'processed', 'feature_columns.json')


@click.group()
def cli():
    """A CLI for the 5G Adversarial IDS."""
    pass


@cli.command()
@click.option('--model', 'model_path', required=True, type=click.Path(exists=True), help="Path to the trained model file (.pt).")
@click.option('--data', 'data_path', required=True, type=click.Path(exists=True), help="Path to the pre-scaled test data CSV.")
@click.option('--columns', 'columns_path', required=True, type=click.Path(exists=True), help="Path to the feature columns JSON file.")
def evaluate(model_path: str, data_path: str, columns_path: str):
    """
    Run a raw evaluation of a model against a dataset and print the metrics.
    This command now correctly passes data directly to the evaluator without a scaler.
    """
    click.echo(f"Running evaluation of '{data_path}' with model '{model_path}'...")
    
    # --- DEFINITIVE FIX ---
    # The `scaler_path` argument is removed, as the new `run_evaluation`
    # correctly assumes the data is pre-scaled. This fixes the double-scaling bug.
    metrics = run_evaluation(
        model_path=model_path,
        data_path=data_path,
        columns_path=columns_path
    )
    # --- END FIX ---
    
    if metrics:
        click.echo(json.dumps(metrics, indent=4))
    else:
        click.secho("Evaluation failed. Please check the error messages above.", fg='red')


@cli.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help="Path to the RAW, unscaled network traffic CSV.")
def assess(dataset: str):
    """
    Run a high-level security assessment on a raw dataset.
    This command handles the entire pipeline: preprocess -> predict -> verdict.
    """
    click.secho("--- 5G IDS Network Assessment Tool ---", bold=True)
    click.echo()

    # Check for required model files first
    if not all([os.path.exists(ROBUST_MODEL_PATH), os.path.exists(SCALER_PATH), os.path.exists(COLUMNS_PATH)]):
        click.secho("ERROR: Core model files not found!", fg='red')
        click.echo("Please ensure 'models/robust_model.pt', 'data/processed/scaler.joblib', and 'data/processed/feature_columns.json' exist.")
        click.echo("You must run the preprocessing and training scripts first.")
        return

    # Create a temporary directory to store the processed version of the user's data
    temp_dir = tempfile.mkdtemp()
    
    try:
        click.echo(f"[1/2] Analyzing traffic from '{os.path.basename(dataset)}'...")
        
        # Preprocess the raw data in the temporary directory
        # This will scale the data and save it as a temporary file
        processed_file_path = os.path.join(temp_dir, 'temp_processed.csv')
        success = preprocess_data(dataset, temp_dir, SCALER_PATH, COLUMNS_PATH, save_path=processed_file_path)

        if not success:
            click.secho("Analysis failed during data preprocessing.", fg='red')
            return

        click.echo("Analysis complete.")
        click.echo()
        click.echo("[2/2] Generating Security Verdict...")
        click.echo()

        # Run evaluation on the temporarily processed file
        # --- DEFINITIVE FIX ---
        # The `scaler_path` argument is removed here as well for consistency.
        metrics = run_evaluation(
            model_path=ROBUST_MODEL_PATH,
            data_path=processed_file_path,
            columns_path=COLUMNS_PATH
        )
        # --- END FIX ---

        if not metrics:
            click.secho("Could not generate verdict due to an evaluation error.", fg='red')
            return

        f1 = metrics['f1']
        precision = metrics['precision']
        recall = metrics['recall']
        
        click.secho("--- Assessment Metrics ---", bold=True)
        click.echo(f"F1-Score: {f1:.4f}")
        click.echo(f"Precision: {precision:.4f}")
        click.echo(f"Recall: {recall:.4f}")
        click.echo()
        click.secho("--- VERDICT ---", bold=True)

        if f1 < 0.4:
            click.secho("STATUS: COMPROMISED.", fg='red', bold=True)
            click.echo("The model detected significant anomalies consistent with adversarial attacks. Immediate investigation is required.")
        elif f1 < 0.6:
            click.secho("STATUS: CAUTION.", fg='yellow', bold=True)
            click.echo("The model performance is degraded, suggesting potential evasion or novel threats. Manual review is advised.")
        else:
            click.secho("STATUS: SECURE.", fg='green', bold=True)
            click.echo("The model shows strong performance. Traffic patterns appear normal and the network exhibits resilience.")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    cli()

