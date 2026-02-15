"""
2-Classification.py

This script performs classification using Random Forest, XGBoost, and MLP 
   Neural Network classifiers on a dataset. 
It evaluates the models using confusion matrices and various metrics, and 
   compares their performance against a baseline model (IOC_Mod). 
The results are saved to CSV and text files for further analysis.

"""

import os
import warnings
import importlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pandas as pd
import FunEvalClass as FEvalClass

# -----------------------------------------------------------
# Initial settings
# -----------------------------------------------------------
importlib.reload(FEvalClass)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 0)

show_res = True     # Show on screen and terminal

# Create a results/classification directory if it doesn't exist
DIR_RESULTS = 'results/classification'
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)


# Version definition
# ---------------------------------------------------------------------------------------------------
# Format of the name of the version: ORIGIN-SPLIT-BALANCE-DIA_PREV-SEM_PREV_TRAIN_MATCH_TEST
# ---------------------------------------------------------------------------------------------------
v = "ORI-11_19-UNBAL-DEL-MATCH"
comment = "DS SOURCE = ORI..: Original DS 383 rows\n" \
          "SPLIT     = 11_19: Manual, days 11 to 19 Nov 2023 testing (12 rows with IOC_Obs=1) and the rest for training\n" \
          "                   (371, 361 IOC_Obs=0 + 10 IOC_Obs=1).\n" \
          "BALANCE   = UNBAL: Originally unbalanced, 361 IOC_Obs=0 + 10 IOC_Obs=1 in training set, 12 IOC_Obs=1 in test set.\n" \
          "DIA_PREV  = DEL..: Deleted due to sem_prev is more granular than dia_prev and can capture more temporal patterns.\n" \
          "SEM_PREV  = MATCH: Rows in training set was removed where sem_prev is not in the test set sem_prev values,\n" \
          "                   to ensure the model learns from the same temporal distribution as the test set.\n"
print(f"Version: {v}\n\n{comment}")

# -----------------------------------------------------------------------------------------------------------------
# BAL-TOT-11-19     361 (IOC_Obs=0) x 22 (IOC_Obs=1) transformed to 361 x 361 balanced dataset using SMOTE,
# v = "BAL-TOT-11_19"
#                   then split manually in a manner that days 11-19 of Nov 2023
#
# 361 x 22 Dataset splited manually in a manner that days 11 to 19 of November 2023 was
#                   used for testing and the rest for training. Using balanced dataset with SMOTE: 361 (IOC_Obs=0) x 361 (IOC_Obs=1)
# -----------------------------------------------------------------------------------------------------------------

# Load the full dataset
DS = pd.read_csv("1.1-DATA_ORIGINAL-FULL.csv")

# Remove unnecessary columns for any classification version
DS.drop(columns=['data_rod', 'Tx_Obs', "IOC_Mod",
        "IOC_Obs"], inplace=True)
print(DS.columns)

if v == "ORI-11_19-UNBAL-DEL-MATCH":
    print(f"Version: {v}\n\n{comment}")

    # SPLIT the dataset into training and testing sets in a manner that testing lies between 20231111 and 20231119
    test_mask = (DS['data_prev'] >= 20231111) & (DS['data_prev'] <= 20231119)
    X_train = DS[~test_mask]
    y_train = DS[~test_mask]['OC_Obs_Candid'].values
    X_test = DS[test_mask]
    y_test = DS[test_mask]['OC_Obs_Candid'].values

    # BALANCE: Originally unbalanced, 361 IOC_Obs=0 + 10 IOC_Obs=1 in training set, 12 IOC_Obs=1 in test set.
    # Nothing to do here, we will train the models with the original unbalanced dataset to see how they perform in this real-world scenario.

    # DIA_PREV: Deleted due to sem_prev is more granular than dia_prev and can capture more temporal patterns.
    X_train.drop(columns=['dia_prev'], inplace=True)
    X_test.drop(columns=['dia_prev'], inplace=True)

    # SEM_PREV: Remove from X_train all rows where sem_prev is not in the X_test sem_prev values
    X_train = X_train[X_train['sem_prev'].isin(X_test['sem_prev'].unique())]
    y_train = y_train[X_train.index]


# -------------------------------------------------------------------
# From now on, common treatment for all versions
# -------------------------------------------------------------------

# Delete columns not used for classification from both training and testing sets.:
# 'data_prev" due to it's not a feature but a temporal identifier for splitting the dataset into training and testing sets.
# 'OC_Obs_Candid' due to it's used as the target variable.
X_train.drop(columns=['data_prev', 'OC_Obs_Candid'], inplace=True)
X_test.drop(columns=['data_prev', 'OC_Obs_Candid'], inplace=True)

print("* Number of rows of the training and testing sets.")
print(len(X_train), len(X_test))

print("* Distribution of sem_prev values for the training and testing sets.")
print(X_train['sem_prev'].value_counts().sort_index().to_frame().T)
print(X_test['sem_prev'].value_counts().sort_index().to_frame().T)

print("* Predictors:", X_train.columns.tolist())
print("* Target:", "OC_Obs_Candid")


# Automatic split aborted due to the specific requirement of using the whole month of 2023-11-03 as the test set.
# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=42, stratify=y)

# print("Check the distribution of the target variable in the training and testing sets")
# print(y_test.value_counts())
# print(y_train.value_counts())
# The test is 100% inbalanced due to IOC_Obs for the requirements is True for the whole period!


# ================================================
# 2.1) RANDOM FOREST CLASSIFIER - MODEL EVALUATION
# ================================================
# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
rf_pred = rf_classifier.predict(X_test)

# ------------------------------------------------
# EVALUATING THE RANDOM FOREST CLASSIFIER
# ------------------------------------------------
# Confusion Matrix
rf_cm = FEvalClass.eval_confusion_matrix(
    y_test, rf_pred, 'Blues', 'Random Forest Classifier', 'RF', v, show_res)

# Metrics
rf_metrics = FEvalClass.eval_metrics(
    y_test, rf_pred, 'Random Forest Classifier', 'RF', v, show_res)

# Feature importance
rf_importances = FEvalClass.eval_feature_importance(
    rf_classifier, X_train.columns, 'Random Forest Classifier', 'RF', v, show_res)


# ==========================================
# 2.2) XGBOOST CLASSIFIER - MODEL EVALUATION
# ==========================================

# Create and train XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

# Train the model
xgb_classifier.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_classifier.predict(X_test)

# ------------------------------------------------
# EVALUATING THE XGBOOST CLASSIFIER
# ------------------------------------------------
# Confusion Matrix
xgb_cm = FEvalClass.eval_confusion_matrix(
    y_test, xgb_pred, 'Greens', 'XGBoost Classifier', 'XGB', v, show_res)

# Metrics
xgb_metrics = FEvalClass.eval_metrics(
    y_test, xgb_pred, 'XGBoost Classifier', 'XGB', v, show_res)

# Feature importance
xgb_importances = FEvalClass.eval_feature_importance(
    xgb_classifier, X_train.columns, 'XGBoost Classifier', 'XGB', v, show_res)


# =====================================================
# 2.3) MLP NEURAL NETWORK CLASSIFIER - MODEL EVALUATION
# =====================================================

# Standardize features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build an efficient neural network using scikit-learn MLPClassifier
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    batch_size=32,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    alpha=0.001,  # L2 regularization
    random_state=42,
    verbose=False
)

# Train the model
mlp_classifier.fit(X_train_scaled, y_train)

# Make predictions
mlp_pred = mlp_classifier.predict(X_test_scaled)


# ------------------------------------------------
# EVALUATING THE MLP NEURAL NETWORK CLASSIFIER
# ------------------------------------------------
# Confusion Matrix
mlp_cm = FEvalClass.eval_confusion_matrix(
    y_test, mlp_pred, 'Oranges', 'MLP Neural Network Classifier', 'MLP', v, show_res)

# Metrics
mlp_metrics = FEvalClass.eval_metrics(
    y_test, mlp_pred, 'MLP Neural Network Classifier', 'MLP', v, show_res)

# Feature importance using permutation importance (for neural networks, traditional feature importance is not available)
FEvalClass.eval_mlp_feature_importance(
    mlp_classifier, X_test_scaled, y_test, X_train, v, show_res)

# MLP Training history is not available in scikit-learn's MLPClassifier, so we cannot plot it.
# For that, we would need to use a deep learning framework like TensorFlow or PyTorch.
FEvalClass.plot_mlp_training_history(mlp_classifier, v, show_res)

# =====================================================
# 2.4) Evaluate IOC_Mod against IOC_Obs as a baseline
# =====================================================
# Confusion Matrix
eta_cm = FEvalClass.eval_confusion_matrix(
    y_test, X_test['IOC_Mod'], 'Purples', 'Baseline Model (eta)', 'ETA', v, show_res)

# Metrics
eta_metrics = FEvalClass.eval_metrics(
    y_test, X_test['IOC_Mod'], 'Baseline Model (eta)', 'ETA', v, show_res)

# Feature importance is not applicable for the baseline model since it's just a single feature (IOC_Mod).

# =====================================================
# MODEL COMPARISON: ALL 4 CLASSIFIERS
# =====================================================
# Append the predictions to the test set and save to CSV
results = X_test.copy()
results["RF_IOC"] = rf_pred
results["XGB_IOC"] = xgb_pred
results["MLP_IOC"] = mlp_pred
results["IOC_Obs"] = y_test
results.to_csv(
    f'{DIR_RESULTS}/00-RESULTS-Classification_predictions_{v}.csv', index=False)


# ============================================
# Model Comparison: All 4 Classifiers
# Use accuracy, precision, recall, and F1-score to compare the models in a tabular format.
# Explain what these metrics mean in the context of our problem and which one we prioritize (e.g., F1-score for imbalanced classes).
# ============================================
LINE_LEN = 60
res_text = "\n" + "=" * LINE_LEN + "\n"
res_text += "MODEL COMPARISON: IOC_Mod vs RF vs XGB vs MLP\n"
res_text += "-" * LINE_LEN + "\n"
res_text += f"{'Metric':<15}{'IOC_Mod':>10}{'RF':>10}{'XGB':>10}{'MLP':>10}\n"  # nopep8
res_text += f"{'Accuracy':<15}{eta_metrics['ETA_accuracy']*100:10.2f}%{rf_metrics['RF_accuracy']*100:9.2f}%{xgb_metrics['XGB_accuracy']*100:9.2f}%{mlp_metrics['MLP_accuracy']*100:9.2f}%\n"  # nopep8
res_text += f"{'Precision':<15}{eta_metrics['ETA_precision']:10.4f}{rf_metrics['RF_precision']:10.4f}{xgb_metrics['XGB_precision']:10.4f}{mlp_metrics['MLP_precision']:10.4f}\n"  # nopep8
res_text += f"{'Recall':<15}{eta_metrics['ETA_recall']:10.4f}{rf_metrics['RF_recall']:10.4f}{xgb_metrics['XGB_recall']:10.4f}{mlp_metrics['MLP_recall']:10.4f}\n"  # nopep8
res_text += f"{'F1-Score':<15}{eta_metrics['ETA_f1_score']:10.4f}{rf_metrics['RF_f1_score']:10.4f}{xgb_metrics['XGB_f1_score']:10.4f}{mlp_metrics['MLP_f1_score']:10.4f}\n"  # nopep8
res_text += "=" * LINE_LEN + "\n"

# Explain what these metrics mean in the context of our problem and which one we prioritize (e.g., F1-score for imbalanced classes).
res_text += "- Accuracy: The proportion of correct predictions (both true positives and true negatives) out of all predictions. Care if imbalanced DS.\n"
res_text += "- Precision: The proportion of true positives out of all positive predictions. How many of the predicted positive cases were actually positive.\n"
res_text += "- Recall: The proportion of true positives out of all actual positives. How many of the actual positive cases were correctly identified by the model.\n"
res_text += "- F1-Score: The harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. More useful DS is imbalanced.\n"

if show_res:
    print(res_text)

# Save the results to a text file
with open(f'{DIR_RESULTS}/00-RESULTS-Model_Comparison_{v}.txt', 'w', encoding='utf-8') as f:
    f.write(res_text)

# Determine better model among all 4 (including baseline)
scores = {"Eta": eta_metrics['ETA_f1_score'], "RF": rf_metrics['RF_f1_score'],
          "XGB": xgb_metrics['XGB_f1_score'], "MLP": mlp_metrics['MLP_f1_score']}
best_model = max(scores, key=scores.get)
print(
    f"\n🏆 Best Model (by F1-Score): {best_model} (F1 = {scores[best_model]:.4f})")
