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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
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
# Format of the name of the version: ORIGIN-SPLIT-BALANCE-SEM_PREV_TRAIN_MATCH_TEST
# ---------------------------------------------------------------------------------------------------
# v = "01-DS=Ori,SP=11_19,BAL=No,SP=MatchTrainTest"
# comment = "Simulate a real-world scenario and make predictions for the exact period in which the event occurred.\n" \
#           "DS  (DATA SOURCE) = ORI..: Original DS 383 rows\n" \
#           "SP  (SPLIT)       = 11_19: Manual, days 11 to 19 Nov 2023 testing (12 rows with IOC_Obs=1) and the rest for training\n" \
#           "                           (371, 361 IOC_Obs=0 + 10 IOC_Obs=1).\n" \
#           "BAL (BALANCE)     = UNBAL: Originally unbalanced, 361 IOC_Obs=0 + 10 IOC_Obs=1 in training set, 12 IOC_Obs=1 in test set.\n" \
#           "SP  (SEM_PREV)    = MATCH: Rows in training set was removed where sem_prev is not in the test set sem_prev values,\n" \
#           "                           to ensure the model learns from the same temporal distribution as the test set.\n"
# ---------------------------------------------------------------------------------------------------
# v = "02-DS=Ori+Bal,SP=11_19,BAL=TrainAndTest,SP=MatchTrainTest"
# comment = "Test the use of balanced training and test sets.\n" \
#           "DS  (DATA SOURCE) = Ori+Bal: Original DS 383 rows is balanced.\n" \
#           "SP  (SPLIT)       = 11_19: Manual, days 11 to 19 Nov 2023 testing (12 rows with IOC_Obs=1) and the rest for training\n" \
#           "                           (371, 361 IOC_Obs=0 + 10 IOC_Obs=1).\n" \
#           "BAL (BALANCE)     = TrainAndTest: Both train and test sets are balanced using SMOTE.\n" \
#           "SP  (SEM_PREV)    = MATCH: Rows in training set was removed where sem_prev is not in the test set sem_prev values,\n" \
#           "                           to ensure the model learns from the same temporal distribution as the test set.\n"
# ---------------------------------------------------------------------------------------------------
# v = "03-DS=Ori,SP=70-30,BAL=None,SP=MatchTrainTest"
# comment = "Test the automatic split in the original dataset.\n" \
#           "DS  (DATA SOURCE) = Ori: Original DS 383 rows.\n" \
#           "SP  (SPLIT)       = 70-30: Automatic split of 70% training and 30% testing.\n" \
#           "BAL (BALANCE)     = None.\n" \
#           "SP  (SEM_PREV)    = MATCH: Rows in training set was removed where sem_prev is not in the test set sem_prev values,\n" \
#           "                           to ensure the model learns from the same temporal distribution as the test set.\n"
# ---------------------------------------------------------------------------------------------------
v = "04-DS=Ori,SP=70-30,BAL=Balanced,SP=MatchTrainTest"
comment = "Test the automatic balance and split in the original dataset.\n" \
          "DS  (DATA SOURCE) = Ori: Original DS 383 rows balanced.\n" \
          "SP  (SPLIT)       = 70-30: Automatic split of 70% training and 30% testing.\n" \
          "BAL (BALANCE)     = Balanced .\n" \
          "SP  (SEM_PREV)    = MATCH: Rows in training set was removed where sem_prev is not in the test set sem_prev values,\n" \
          "                           to ensure the model learns from the same temporal distribution as the test set.\n"

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
PREDICTORS = ['sem_prev', 'tsfc', 'magv', 'ur2m', 'Tx_Mod',
              'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod', 'Cl_Tx_Obs', 'OC_Mod_Candid', 'IOC_Mod']
TARGET = 'OC_Obs_Candid'

# Remove unnecessary columns for any classification version
DS.drop(columns=['data_rod', "dia_prev", 'Tx_Obs', "IOC_Obs"], inplace=True)
print(DS.columns)

# Check the distribution of the target variable in the original dataset
print("* Original dataset class distribution:",
      FEvalClass.count_classes(DS[TARGET].values))

# ----------------------------------------------------------------------------------------------
if v == "01-DS=Ori,SP=11_19,BAL=No,SP=MatchTrainTest":
    # ------------------------------------------------------------------------------------------

    print(f"Version: {v}\n\n{comment}")
    DIR_VERSION = f'{DIR_RESULTS}/v_{v}'
    if not os.path.exists(DIR_VERSION):
        os.makedirs(DIR_VERSION)

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

# ----------------------------------------------------------------------------------------------
if v == "02-DS=Ori+Bal,SP=11_19,BAL=TrainAndTest,SP=MatchTrainTest":
    # ------------------------------------------------------------------------------------------
    print(f"Version: {v}\n\n{comment}")
    DIR_VERSION = f'{DIR_RESULTS}/v_{v}'
    if not os.path.exists(DIR_VERSION):
        os.makedirs(DIR_VERSION)

    print("* Balancing the original dataset using SMOTE before splitting into training and testing sets...")
    features_for_balance = DS.drop(columns=['OC_Obs_Candid'])
    target_for_balance = DS['OC_Obs_Candid']
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(
        features_for_balance, target_for_balance)
    data_balanced = X_balanced.copy()
    data_balanced['OC_Obs_Candid'] = y_balanced
    print("* Balanced dataset class distribution:",
          FEvalClass.count_classes(y_balanced.values))
    # 314 x 314

    print("* Split the balanced dataset into training and testing sets in a manner that testing lies between 20231111 and 20231119")
    test_mask = (data_balanced['data_prev'] >= 20231111) & (
        data_balanced['data_prev'] <= 20231119)
    X_train = data_balanced[~test_mask]
    y_train = data_balanced[~test_mask]['OC_Obs_Candid'].values
    X_test = data_balanced[test_mask]
    y_test = data_balanced[test_mask]['OC_Obs_Candid'].values

    # Check the distribution of the target variable in the training and testing sets after balancing
    print("* Training set class distribution:",
          FEvalClass.count_classes(y_train))
    print("* Testing set class distribution:",
          FEvalClass.count_classes(y_test))

    print("* Now, rebalancing the training set again, since various rows with OC_Obs_Candid=1 were moved to the testing set.")
    smote_train = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote_train.fit_resample(
        X_train.drop(columns=['OC_Obs_Candid']), y_train)
    X_train_balanced['OC_Obs_Candid'] = y_train_balanced
    X_train = X_train_balanced.copy()
    y_train = X_train['OC_Obs_Candid'].values

    print("* Training set class distribution:",
          FEvalClass.count_classes(y_train))
    print("* Testing set class distribution:",
          FEvalClass.count_classes(y_test))

# ----------------------------------------------------------------------------------------------
if v == "03-DS=Ori,SP=70-30,BAL=None,SP=MatchTrainTest":
    # ------------------------------------------------------------------------------------------
    print(f"Version: {v}\n\n{comment}")
    DIR_VERSION = f'{DIR_RESULTS}/v_{v}'
    if not os.path.exists(DIR_VERSION):
        os.makedirs(DIR_VERSION)

    print("* Split the original dataset into training and testing sets in 70% and 30%.")
    X_train, X_test, y_train, y_test = train_test_split(
        DS, DS['OC_Obs_Candid'], test_size=0.3, random_state=42, stratify=DS['OC_Obs_Candid'])
    # Check the distribution of the target variable in the training and testing sets after balancing
    print("* Training set class distribution:",
          FEvalClass.count_classes(y_train))
    print("* Testing set class distribution:",
          FEvalClass.count_classes(y_test))


# ----------------------------------------------------------------------------------------------
if v == "04-DS=Ori,SP=70-30,BAL=Balanced,SP=MatchTrainTest":
    # ------------------------------------------------------------------------------------------
    print(f"Version: {v}\n\n{comment}")
    DIR_VERSION = f'{DIR_RESULTS}/v_{v}'
    if not os.path.exists(DIR_VERSION):
        os.makedirs(DIR_VERSION)

    print("* Balancing the original dataset using SMOTE before splitting into training and testing sets...")
    smote = SMOTE(random_state=42)
    features_for_balance = DS.drop(columns=['OC_Obs_Candid'])
    target_for_balance = DS['OC_Obs_Candid']
    X_balanced, y_balanced = smote.fit_resample(
        features_for_balance, target_for_balance)
    data_balanced = X_balanced.copy()
    data_balanced['OC_Obs_Candid'] = y_balanced
    print("* Balanced dataset class distribution:",
          FEvalClass.count_classes(y_balanced.values))
    # 314 x 314

    print("* Split the original dataset into training and testing sets in 70% and 30%.")
    X_train, X_test, y_train, y_test = train_test_split(
        DS, DS['OC_Obs_Candid'], test_size=0.3, random_state=42, stratify=DS['OC_Obs_Candid'])
    # Check the distribution of the target variable in the training and testing sets after balancing
    print("* Training set class distribution:",
          FEvalClass.count_classes(y_train))
    print("* Testing set class distribution:",
          FEvalClass.count_classes(y_test))

    # Check the distribution of the target variable in the training and testing sets after balancing
    print("* Training set class distribution:",
          FEvalClass.count_classes(y_train))
    print("* Testing set class distribution:",
          FEvalClass.count_classes(y_test))


# -------------------------------------------------------------------
# From now on, common treatment for all versions
# -------------------------------------------------------------------

# Save a txt file with the version description
with open(f'{DIR_RESULTS}/v_{v}.txt', 'w', encoding='utf-8') as f:
    x = f.write(comment)

# SEM_PREV: Remove from X_train all rows where sem_prev is not in the X_test sem_prev values
X_train = X_train[X_train['sem_prev'].isin(X_test['sem_prev'].unique())]
y_train = y_train[X_train.index]
print("* Distribution of sem_prev values for the training and testing sets.")
print(X_train['sem_prev'].value_counts().sort_index().to_frame().T)
print(X_test['sem_prev'].value_counts().sort_index().to_frame().T)

# Delete columns not used for classification from both training and testing sets.:
# 'data_prev" due to it's not a feature but a temporal identifier for splitting the dataset into training and testing sets.
# 'OC_Obs_Candid' due to it's used as the target variable.
X_train.drop(columns=['data_prev', 'OC_Obs_Candid'], inplace=True)
X_test.drop(columns=['data_prev', 'OC_Obs_Candid'], inplace=True)

print("* Number of rows of the training and testing sets.")
print(len(X_train), len(X_test))

print("* Predictors:", X_train.columns.tolist())
print("* Target:", TARGET)
# Append to the file with the predictors and target information
with open(f'{DIR_RESULTS}/v_{v}.txt', 'a', encoding='utf-8') as f:
    x = f.write(f"\n* Predictors: {X_train.columns.tolist()}\n")
    x = f.write(f"* Target: {TARGET}\n")


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

# Standardize features between 0 and 1 for neural networks, which are sensitive to feature scales.
# We will use MinMaxScaler to scale the features to the range [0, 1].
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# pd.DataFrame(X_train_scaled, columns=X_train.columns).describe()

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
    y_test, X_test['OC_Mod_Candid'], 'Purples', 'Baseline Model (eta)', 'ETA', v, show_res)

# Metrics
eta_metrics = FEvalClass.eval_metrics(
    y_test, X_test['OC_Mod_Candid'], 'Baseline Model (eta)', 'ETA', v, show_res)

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
    f'{DIR_VERSION}/00-RESULTS-Classification_predictions.csv', index=False)


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
with open(f'{DIR_VERSION}/00-RESULTS-Model_Comparison.txt', 'w', encoding='utf-8') as f:
    x = f.write(res_text)

# Determine better model among all 4 (including baseline)
scores = {"Eta": eta_metrics['ETA_f1_score'], "RF": rf_metrics['RF_f1_score'],
          "XGB": xgb_metrics['XGB_f1_score'], "MLP": mlp_metrics['MLP_f1_score']}
best_model = max(scores, key=scores.get)
print(
    f"\n🏆 Best Model (by F1-Score): {best_model} (F1 = {scores[best_model]:.4f})")
