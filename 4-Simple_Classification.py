"""
4-Simple_Classification.py

This script implements a simple but efficient classification algorithm
to classify OC_Obs_Candid using a subset of predictors.

Uses the following predictors:
- sem_prev, tsfc, magv, ur2m, Tx_Mod, P90cl_Tx_Mod, P90cl_Tx_Obs, Cl_Tx_Mod, Cl_Tx_Obs, OC_Mod_Candid

Dataset: 1.1-DATA_ORIGINAL-FULL.csv
Target: OC_Obs_Candid

Models tested: Logistic Regression, Decision Tree, Random Forest
"""

import os
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =========================================
# Configuration
# =========================================
RANDOM_STATE = 42
DIR_RESULTS = 'results/simple_classification'

if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

# Define predictors and target
PREDICTORS = ['sem_prev', 'tsfc', 'magv', 'ur2m', 'Tx_Mod',
              'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod', 'Cl_Tx_Obs', 'OC_Mod_Candid']
TARGET = 'OC_Obs_Candid'

print("=" * 70)
print("SIMPLE CLASSIFICATION: OC_Obs_Candid Prediction")
print("=" * 70)
print(f"\nPredictors: {PREDICTORS}")
print(f"Target: {TARGET}\n")

# =========================================
# Load and Prepare Data
# =========================================
print("Loading data...")
df = pd.read_csv("1.1-DATA_ORIGINAL-FULL.csv")

print(f"Full dataset shape: {df.shape}")
print(f"\nDataset columns: {df.columns.tolist()}")

# =========================================
# Manual Train-Test Split (Date-based)
# =========================================
print("\n" + "=" * 70)
print("MANUAL TRAIN-TEST SPLIT (Date-based)")
print("=" * 70)

# Split based on data_prev: test set = 20231111 to 20231119, training set = rest
test_mask = (df['data_prev'] >= 20231111) & (df['data_prev'] <= 20231119)
train_mask = ~test_mask

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

print(f"\nTest set date range: 20231111 to 20231119")
print(f"Training set size: {len(df_train)}")
print(f"Test set size: {len(df_test)}")

# =========================================
# Validate sem_prev consistency
# =========================================
print("\n" + "=" * 70)
print("VALIDATING SEM_PREV CONSISTENCY")
print("=" * 70)

sem_prev_train = set(df_train['sem_prev'].unique())
sem_prev_test = set(df_test['sem_prev'].unique())

print(f"\nUnique sem_prev values in training set: {sorted(sem_prev_train)}")
print(f"Unique sem_prev values in test set: {sorted(sem_prev_test)}")

# Assertion: training set should not have sem_prev values not in test set
sem_prev_only_in_train = sem_prev_train - sem_prev_test
if sem_prev_only_in_train:
    print(
        f"\n⚠️  WARNING: Found sem_prev values in training set not in test set: {sem_prev_only_in_train}")
    print("Removing these rows from training set to ensure consistency...")
    df_train = df_train[df_train['sem_prev'].isin(sem_prev_test)]
    print(f"Training set size after filtering: {len(df_train)}")
else:
    print("\n✓ Training set sem_prev values are consistent with test set.")

# Extract X and y
X_train = df_train[PREDICTORS].copy()
y_train = df_train[TARGET].copy()
X_test = df_test[PREDICTORS].copy()
y_test = df_test[TARGET].copy()

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"\nTraining set target distribution:\n{y_train.value_counts()}")
print(f"\nTest set target distribution:\n{y_test.value_counts()}")

# Check for missing values
print(f"\nMissing values in training predictors:\n{X_train.isnull().sum()}")
print(f"\nMissing values in test predictors:\n{X_test.isnull().sum()}\n")

# =========================================
# Feature Scaling
# =========================================
print("Scaling features between 0 and 1...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# =========================================
# Model 1: Logistic Regression
# =========================================
print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

lr_classifier = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver='lbfgs'
)

lr_classifier.fit(X_train_scaled, y_train)
lr_pred = lr_classifier.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")

# =========================================
# Model 2: Decision Tree
# =========================================
print("\n" + "=" * 70)
print("MODEL 2: DECISION TREE")
print("=" * 70)

dt_classifier = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1-Score:  {dt_f1:.4f}")

# =========================================
# Model 3: Random Forest
# =========================================
print("\n" + "=" * 70)
print("MODEL 3: RANDOM FOREST")
print("=" * 70)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")

# =========================================
# Model Comparison
# =========================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [lr_accuracy, dt_accuracy, rf_accuracy],
    'Precision': [lr_precision, dt_precision, rf_precision],
    'Recall': [lr_recall, dt_recall, rf_recall],
    'F1-Score': [lr_f1, dt_f1, rf_f1]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Find best model
best_idx = comparison_df['F1-Score'].idxmax()
best_model = comparison_df.loc[best_idx, 'Model']
best_f1 = comparison_df.loc[best_idx, 'F1-Score']

print(f"\n🏆 Best Model (by F1-Score): {best_model} (F1 = {best_f1:.4f})")

# Save comparison to CSV
comparison_df.to_csv(os.path.join(
    DIR_RESULTS, '00-RESULTS-Model_Comparison.csv'), index=False)

# =========================================
# Confusion Matrices Visualization
# =========================================
print("\n" + "=" * 70)
print("GENERATING CONFUSION MATRICES")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    ('Logistic Regression', lr_pred, 'Blues'),
    ('Decision Tree', dt_pred, 'Greens'),
    ('Random Forest', rf_pred, 'Oranges')
]

for (name, pred, cmap), ax in zip(models, axes):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                cbar=True, square=True, annot_kws={'size': 14})
    ax.set_title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_xticklabels(['No IOC', 'IOC'])
    ax.set_yticklabels(['No IOC', 'IOC'])

plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTS, 'Confusion_Matrices.png'),
            dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved to: Confusion_Matrices.png")
plt.close()

# =========================================
# Detailed Results Report
# =========================================
print("\n" + "=" * 70)
print("GENERATING DETAILED REPORT")
print("=" * 70)

report_text = "=" * 70 + "\n"
report_text += "SIMPLE CLASSIFICATION RESULTS\n"
report_text += "=" * 70 + "\n\n"

report_text += "DATASET INFORMATION\n"
report_text += "-" * 70 + "\n"
report_text += f"Total samples: {len(df)}\n"
report_text += f"Training samples: {len(X_train)}\n"
report_text += f"Test samples: {len(X_test)}\n"
report_text += f"Split method: Manual date-based (test: 2023-11-11 to 2023-11-19, training: rest)\n"
report_text += f"\nPredictors ({len(PREDICTORS)}): {', '.join(PREDICTORS)}\n"
report_text += f"Target: {TARGET}\n"
report_text += f"\nTraining target distribution:\n{y_train.value_counts().to_string()}\n"
report_text += f"Test target distribution:\n{y_test.value_counts().to_string()}\n"
report_text += "\n"

# Detailed results for each model
report_text += "=" * 70 + "\n"
report_text += "MODEL RESULTS\n"
report_text += "=" * 70 + "\n\n"

models_info = [
    ('LOGISTIC REGRESSION', lr_accuracy, lr_precision, lr_recall, lr_f1, lr_pred),
    ('DECISION TREE', dt_accuracy, dt_precision, dt_recall, dt_f1, dt_pred),
    ('RANDOM FOREST', rf_accuracy, rf_precision, rf_recall, rf_f1, rf_pred)
]

for model_name, acc, prec, rec, f1, pred in models_info:
    report_text += f"\n{model_name}\n"
    report_text += "-" * 70 + "\n"
    report_text += f"Accuracy:  {acc:.4f}\n"
    report_text += f"Precision: {prec:.4f}\n"
    report_text += f"Recall:    {rec:.4f}\n"
    report_text += f"F1-Score:  {f1:.4f}\n"
    report_text += "\nConfusion Matrix:\n"
    cm = confusion_matrix(y_test, pred)
    report_text += f"  TN: {cm[0, 0]:4d}  |  FP: {cm[0, 1]:4d}\n"
    report_text += f"  FN: {cm[1, 0]:4d}  |  TP: {cm[1, 1]:4d}\n"
    report_text += "\nClassification Report:\n"
    report_text += classification_report(y_test, pred)
    report_text += "\n"

report_text += "=" * 70 + "\n"
report_text += "MODEL COMPARISON SUMMARY\n"
report_text += "=" * 70 + "\n"
report_text += comparison_df.to_string(index=False) + "\n\n"
report_text += f"Best Model (by F1-Score): {best_model} (F1 = {best_f1:.4f})\n"
report_text += "=" * 70 + "\n"

# Save report
with open(os.path.join(DIR_RESULTS, '00-RESULTS-Classification_Report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print("✓ Detailed report saved to: 00-RESULTS-Classification_Report.txt")
print("\n" + report_text)

print("\n✓ All results saved to:", DIR_RESULTS)
