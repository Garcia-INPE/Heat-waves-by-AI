"""
Script to compare OC_XGB_Candidate and OC_Mod_Candid performance
using OC_Obs_Candid as ground truth
"""
from sklearn.metrics import ConfusionMatrixDisplay
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import FunPlot as FPlot
importlib.reload(FPlot)

pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 0)

DIR_INPUT = 'output/3.0-regression/02-PRED'
DIR_RESULTS = 'output/3.2-regression_comparison_by_version'
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

# AQUIIIIIII
# Load the data for the version we want to analyze (e.g., v05)
v = "v05"
df = pd.read_csv(os.path.join(DIR_INPUT, f'PRED-{v}.csv'))

print("=" * 80)
print("DATA LOADING AND PROCESSING")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print("First few rows:")
print(df.head())

# Create the new column OC_XGB_Candid
df['OC_XGB_Candid'] = (df['XGB_Pred'] >= df['P90cl_Tx_Obs']).astype(int)

print("Column creation check:")
print(f"OC_XGB_Candid value counts:\n{df['OC_XGB_Candid'].value_counts()}")
print(f"\nOC_Mod_Candid value counts:\n{df['OC_Mod_Candid'].value_counts()}")
print(
    f"\nOC_Obs_Candid value counts (ground truth):\n{df['OC_Obs_Candid'].value_counts()}")

# Extract the three columns of interest
ground_truth = df['OC_Obs_Candid'].values
xgb_candidate = df['OC_XGB_Candid'].values
mod_candidate = df['OC_Mod_Candid'].values

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

# Helper function to compute and display metrics


def compute_metrics(predictions, ground_truth, model_name):
    """Compute and print performance metrics"""
    print(f"\n{model_name}")
    print("-" * 40)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")

    # Accuracy
    acc = accuracy_score(ground_truth, predictions)
    print(f"\nAccuracy:  {acc:.4f}")

    # Sensitivity (Recall / TPR)
    sensitivity = recall_score(ground_truth, predictions, zero_division=0)
    print(f"Sensitivity (Recall/TPR): {sensitivity:.4f}")

    # Specificity (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity (TNR): {specificity:.4f}")

    # Precision
    precision = precision_score(ground_truth, predictions, zero_division=0)
    print(f"Precision: {precision:.4f}")

    # F1-Score
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    print(f"F1-Score: {f1:.4f}")

    # Balanced Accuracy
    balanced_acc = (sensitivity + specificity) / 2
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # Other metrics
    try:
        auc = roc_auc_score(ground_truth, predictions)
        print(f"ROC-AUC: {auc:.4f}")
    except:
        print("ROC-AUC: N/A (requires threshold probability)")

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'Balanced_Accuracy': balanced_acc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


# Compute metrics for both models
metrics_xgb = compute_metrics(xgb_candidate, ground_truth, "OC_XGB_Candid")
metrics_mod = compute_metrics(mod_candidate, ground_truth, "OC_Mod_Candid")

# Create comparison summary
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

comparison_df = pd.DataFrame([metrics_xgb, metrics_mod])
print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS (XGB vs MOD)")
print("=" * 80)
improvements = {
    'Accuracy': metrics_xgb['Accuracy'] - metrics_mod['Accuracy'],
    'Sensitivity': metrics_xgb['Sensitivity'] - metrics_mod['Sensitivity'],
    'Specificity': metrics_xgb['Specificity'] - metrics_mod['Specificity'],
    'Precision': metrics_xgb['Precision'] - metrics_mod['Precision'],
    'F1-Score': metrics_xgb['F1-Score'] - metrics_mod['F1-Score'],
    'Balanced_Accuracy': metrics_xgb['Balanced_Accuracy'] - metrics_mod['Balanced_Accuracy'],
}

for metric, improvement in improvements.items():
    direction = "↑ BETTER" if improvement > 0 else "↓ WORSE" if improvement < 0 else "="
    print(f"{metric:20s}: {improvement:+.4f} {direction}")

# Visualization
plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion matrices visualization

cm_xgb = confusion_matrix(ground_truth, xgb_candidate)
cm_mod = confusion_matrix(ground_truth, mod_candidate)

ConfusionMatrixDisplay(cm_xgb, display_labels=['Negative', 'Positive']).plot(
    ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Confusion Matrix - OC_XGB_Candid',
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(False)

ConfusionMatrixDisplay(cm_mod, display_labels=['Negative', 'Positive']).plot(
    ax=axes[0, 1], cmap='Oranges')
axes[0, 1].set_title('Confusion Matrix - OC_Mod_Candid',
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(False)

# Metrics comparison bar chart
metrics_to_plot = ['Accuracy', 'Sensitivity',
                   'Specificity', 'Precision', 'F1-Score']
xgb_values = [metrics_xgb[m] for m in metrics_to_plot]
mod_values = [metrics_mod[m] for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35

axes[1, 0].bar(x - width/2, xgb_values, width, label='XGB',
               alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.7)
axes[1, 0].bar(x + width/2, mod_values, width, label='MOD',
               alpha=0.8, color='coral', edgecolor='black', linewidth=0.7)
axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].set_title('Performance Metrics Comparison',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(
    metrics_to_plot, rotation=45, ha='right', fontsize=9)
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Improvements visualization
improvements_list = list(improvements.values())
colors = ['green' if v > 0 else 'red' for v in improvements_list]
axes[1, 1].barh(list(improvements.keys()), improvements_list,
                color=colors, alpha=0.7, edgecolor='black', linewidth=0.7)
axes[1, 1].set_xlabel('Improvement (XGB - MOD)', fontsize=11)
axes[1, 1].set_title('Performance Improvements (XGB vs MOD)',
                     fontsize=12, fontweight='bold')
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
axes[1, 1].grid(True, alpha=0.3, axis='x')

fig.suptitle('Candidate Performance Comparison',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig_name = f'{DIR_RESULTS}/{v}-3.2.1-regression_candidate_comparison.png'
plt.savefig(fig_name,
            dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as '{fig_name}'")

# Print classification reports
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

print("\nOC_XGB_Candid:")
print(classification_report(ground_truth, xgb_candidate,
      target_names=['Negative', 'Positive']))

print("\nOC_Mod_Candid:")
print(classification_report(ground_truth, mod_candidate,
      target_names=['Negative', 'Positive']))

# Save comparison results to CSV
comparison_df.to_csv(os.path.join(
    DIR_RESULTS, f'{v}-3.2.2-regression_comparison_results.csv'), index=False)

# Save detailed results
detailed_results = df[['data_rod', 'data_prev', 'Tx_Mod', 'P90cl_Tx_Mod', 'XGB_Pred', "Tx_Obs", "P90cl_Tx_Obs",
                       'OC_XGB_Candid', 'OC_Mod_Candid', 'OC_Obs_Candid']].copy()
detailed_results.to_csv(os.path.join(
    DIR_RESULTS, f'{v}-3.2.3-regression_detailed_predictions.csv'), index=False)

# Plot the time series of XGB predictions, Tx_Mod, P90cl_Tx_Obs and Tx_Obs
# Use FunPlot.py style for consistency
fig = FPlot.plot_xgb_candidate_time_series(
    df, date_cols=['data_rod', 'data_prev'], max_ticks=15)
plt.savefig(os.path.join(DIR_RESULTS, f'{v}-3.2.4-regression_time_series.png'),
            dpi=300, bbox_inches='tight')

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
