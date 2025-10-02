import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
import argparse
import os
import json

# We must re-import these classes to use them
from src.baseline import BaselineIDS, PFCPDataset

def set_professional_style():
    """Sets a global style for all plots to ensure consistency and quality."""
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        rc={
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.titleweight': 'bold',
            'font.family': 'serif',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 18,
            'figure.dpi': 300
        }
    )

def plot_metric_curve(curve_type, y_true, y_scores, title, output_path):
    """
    Creates a publication-quality Precision-Recall or ROC AUC curve.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if curve_type == 'pr':
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        area = auc(recall, precision)
        ax.plot(recall, precision, color='#D64541', lw=2.5, label=f'PR Curve (Area = {area:.3f})')
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
    elif curve_type == 'roc':
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='#4B8BBE', lw=2.5, label=f'ROC Curve (Area = {area:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (Area = 0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
    else:
        raise ValueError("curve_type must be 'pr' or 'roc'")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(title, pad=20)
    ax.legend(loc="lower right" if curve_type == 'roc' else "lower left", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {curve_type.upper()} curve to {output_path}")

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """
    Creates a publication-quality confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={"size": 20, "fontweight": 'bold'}, cbar=False)
    
    ax.set_title(title, pad=20)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_ticklabels(['Normal', 'Malicious'])
    ax.yaxis.set_ticklabels(['Normal', 'Malicious'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def plot_metric_drop_bar_chart(metrics_data, title, output_path):
    """
    Creates a grouped bar chart showing the drop in performance metrics.
    """
    df = pd.DataFrame(metrics_data).T
    df.index.name = 'Condition'
    df_melted = df.reset_index().melt(id_vars='Condition', var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Metric', y='Score', hue='Condition', data=df_melted, ax=ax, palette='viridis')

    ax.set_title(title, pad=20)
    ax.set_ylabel('Score')
    ax.set_xlabel('Performance Metric')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Condition', fontsize=12)
    
    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved metric drop chart to {output_path}")

def plot_final_metrics_bar_chart(metrics_data, title, output_path):
    """
    Creates a simple bar chart for the final metrics of a single model.
    """
    df = pd.Series(metrics_data).reset_index()
    df.columns = ['Metric', 'Score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', data=df, ax=ax, palette='mako')

    ax.set_title(title, pad=20)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)

    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved final metrics chart to {output_path}")

def get_predictions_and_scores(model, loader, device):
    """Helper to get all model outputs for a dataset."""
    model.eval()
    all_targets, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # --- DEFINITIVE FIX for scalar outputs on last batch ---
            # Ensure scores is always at least a 1D array to prevent scalars
            # from breaking concatenation or downstream metrics.
            scores = np.atleast_1d(scores)
            
            preds = (scores > 0.5).astype(int)
            
            all_targets.append(target.cpu().numpy())
            all_preds.append(preds)
            all_scores.append(scores)
    
    # Using concatenate is robust and works for lists with one or more arrays.
    return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate all publication-quality visualizations.")
    parser.add_argument('--processed_dir', type=str, required=True, help="Directory with clean data.")
    parser.add_argument('--adversarial_dir', type=str, required=True, help="Directory with adversarial data.")
    parser.add_argument('--baseline_model_path', type=str, required=True, help="Path to the baseline model.")
    parser.add_argument('--robust_model_path', type=str, required=True, help="Path to the robust model.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save figures.")
    
    args = parser.parse_args()
    
    set_professional_style()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu"

    # --- Load Assets ---
    columns_path = os.path.join(args.processed_dir, 'feature_columns.json')
    with open(columns_path, 'r') as f:
        feature_columns = json.load(f)
    input_features = len(feature_columns)

    # Load Models
    baseline_model = BaselineIDS(input_features).to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
    
    robust_model = BaselineIDS(input_features).to(device)
    robust_model.load_state_dict(torch.load(args.robust_model_path, map_location=device))

    # Create DataLoaders
    loaders = {}
    for name, path in [('clean', os.path.join(args.processed_dir, 'test.csv')),
                       ('fgsm', os.path.join(args.adversarial_dir, 'fgsm_adversaries_eps_0.05.csv')),
                       ('pgd', os.path.join(args.adversarial_dir, 'pgd_adversaries_eps_0.05.csv'))]:
        dataset = PFCPDataset(path, columns_path=columns_path)
        loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # --- Generate Predictions & Metrics ---
    results = {}
    models = {'baseline': baseline_model, 'robust': robust_model}
    for model_name, model_instance in models.items():
        results[model_name] = {}
        for data_name, loader in loaders.items():
            y_true, y_pred, y_scores = get_predictions_and_scores(model_instance, loader, device)
            results[model_name][data_name] = {
                'y_true': y_true, 'y_pred': y_pred, 'y_scores': y_scores,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred)
            }

    # === GENERATE ALL REQUESTED PLOTS ===
    print("\n--- Generating Publication Figures ---")

    # 1. Precision-Recall Curve (Baseline, Clean)
    plot_metric_curve('pr', results['baseline']['clean']['y_true'], results['baseline']['clean']['y_scores'],
                      "Precision-Recall Curve (Baseline Model, Clean Data)",
                      os.path.join(args.output_dir, "fig1_pr_curve_baseline_clean.png"))

    # 2. Confusion Matrix (Baseline, Clean)
    plot_confusion_matrix(results['baseline']['clean']['y_true'], results['baseline']['clean']['y_pred'],
                          "Confusion Matrix (Baseline Model, Clean Data)",
                          os.path.join(args.output_dir, "fig2_cm_baseline_clean.png"))

    # 3. ROC AUC Curve (Baseline, Clean)
    plot_metric_curve('roc', results['baseline']['clean']['y_true'], results['baseline']['clean']['y_scores'],
                      "ROC AUC Curve (Baseline Model, Clean Data)",
                      os.path.join(args.output_dir, "fig3_roc_curve_baseline_clean.png"))

    # 4. Drop in Performance Bar Chart (Baseline)
    # --- DEFINITIVE FIX: Extract only scalar metrics for plotting ---
    # The plotting function expects a dict of dicts of scalars, not numpy arrays.
    baseline_metrics_for_plot = {
        'Clean': {k: v for k, v in results['baseline']['clean'].items() if isinstance(v, (int, float))},
        'FGSM Attack': {k: v for k, v in results['baseline']['fgsm'].items() if isinstance(v, (int, float))},
        'PGD Attack': {k: v for k, v in results['baseline']['pgd'].items() if isinstance(v, (int, float))}
    }
    plot_metric_drop_bar_chart(baseline_metrics_for_plot,
                               "Baseline Model Performance Degradation Under Attack",
                               os.path.join(args.output_dir, "fig4_baseline_metric_drop.png"))

    # 5. Confusion Matrix (Robust Defense on PGD)
    plot_confusion_matrix(results['robust']['pgd']['y_true'], results['robust']['pgd']['y_pred'],
                          "Confusion Matrix (Robust Model, PGD Attack)",
                          os.path.join(args.output_dir, "fig5_cm_robust_pgd.png"))

    # 6. Final Metrics Bar Chart (Robust Defense on PGD)
    # --- DEFINITIVE FIX: Extract only scalar metrics for plotting ---
    robust_metrics_for_plot = {k: v for k, v in results['robust']['pgd'].items() if isinstance(v, (int, float))}
    plot_final_metrics_bar_chart(robust_metrics_for_plot,
                                 "Final Performance of Robust Model Under PGD Attack",
                                 os.path.join(args.output_dir, "fig6_final_robust_metrics.png"))
    
    print("\n--- All visualizations generated successfully. ---")

