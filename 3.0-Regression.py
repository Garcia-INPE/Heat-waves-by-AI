"""Regression for IOC_Mod calibration using Tx_Obs."""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import importlib
import Functions as Fun
import FunPlot as FPlot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE

importlib.reload(Fun)
importlib.reload(FPlot)

# ====================================================================================
# Initial settings
# ====================================================================================
pd.set_option('display.max_columns', 0)
show_res = False     # Show on screen and terminal
LINE_LEN = 55
TARGET = 'Tx_Obs'
v_idx = 1

# Work directories
DIR_INPUT = os.path.join('output', '1.0-analise_dos_dados')
DIR_RESULTS = os.path.join('output', '3.0-regression')
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)
DIR_RESULTS_INFO = os.path.join(DIR_RESULTS, '01-INFO')
if not os.path.exists(DIR_RESULTS_INFO):
    os.makedirs(DIR_RESULTS_INFO)
DIR_RESULTS_PRED = os.path.join(DIR_RESULTS, '02-PRED')
if not os.path.exists(DIR_RESULTS_PRED):
    os.makedirs(DIR_RESULTS_PRED)
DIR_RESULTS_CLASSIF = os.path.join(DIR_RESULTS, '03-PRED_CLASSIF')
if not os.path.exists(DIR_RESULTS_CLASSIF):
    os.makedirs(DIR_RESULTS_CLASSIF)
DIR_RESULTS_COMPOSITE = os.path.join(DIR_RESULTS, '04-BINARY_ANALYSIS')
if not os.path.exists(DIR_RESULTS_COMPOSITE):
    os.makedirs(DIR_RESULTS_COMPOSITE)
DIR_RESULTS_TIMESERIES = os.path.join(DIR_RESULTS, '05-TIME_SERIES_ANALYSIS')
if not os.path.exists(DIR_RESULTS_TIMESERIES):
    os.makedirs(DIR_RESULTS_TIMESERIES)


# DataFrame to store predictions for all versions, to be saved at the end of each version loop
# with the version name in the filename, and used for the results plot.
df_eval = pd.DataFrame()

# ==========================================
# 3. Model Initialization
# ==========================================


class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float


# ====================================================================================
# TRAINING AND TESTING DATASET COMPOUND OPTIONS AND COMMENTS
# ====================================================================================
OP_PREDICTORS = ['ALL', 'AD HOC', 'POSIT_CORR', 'POSIT_CORR+SEM_PREV']
OP_PREDICTORS_COMM = {'ALL': 'All predictors',
                      'AD HOC': 'Predictors selected based on domain knowledge',
                      'POSIT_CORR': 'Predictors with positive correlation to Tx_Obs',
                      'POSIT_CORR+SEM_PREV': 'Predictors with positive correlation to Tx_Obs + SEM_PREV'}
OP_SPLIT = ['11_19', '70_30']
OP_SLIT_COMM = {'11_19': 'Manual split, days 11 to 19 Nov 2023 testing (12 rows with IOC_Obs=1) and the rest for training (371, 361 IOC_Obs=0 + 10 IOC_Obs=1).',
                '70_30': 'Automatic split of 70% training and 30% testing.'}
OP_BALANCE = ['No', 'Yes']
OP_BALANCE_COMM = {'Yes': 'Balance the dataset (force IOC_Obs=1 equals to IOC_Obs=0)',
                   'No': 'Use the original unbalanced dataset.'}
OP_SEM_PREV_MATCH = ['Yes', 'No']
OP_SEM_PREV_MATCH_COMM = {'Yes': 'Remove from X_train all rows where sem_prev is not in the X_test sem_prev values, to ensure the model learns from the same temporal distribution as the test set.',
                          'No': 'Do not filter X_train based on sem_prev values in the test set.'}
# ====================================================================================
# The code below will loop through all combinations of the options for predictors, split, balance and sem_prev match,
# to create different versions of the training and testing datasets, train the models and save the results with a
# version name that includes the options used.
# ====================================================================================
# op_pred=OP_PREDICTORS[0]; op_split=OP_SPLIT[0]; op_bal=OP_BALANCE[0]; op_sem=OP_SEM_PREV_MATCH[0]
for op_pred in OP_PREDICTORS:
    for op_split in OP_SPLIT:
        for op_bal in OP_BALANCE:
            for op_sem in OP_SEM_PREV_MATCH:
                v_str = f"{v_idx:02d}"

                # ====================================================================================
                # INITIAL SETTINGS FOR THIS VERSION OF THE DATASET
                # ====================================================================================
                # Load the full dataset for each test
                DS = pd.read_csv(f"{DIR_INPUT}/1.1-DATA_ORIGINAL-FULL.csv")

                # Start building the comment for this version of the dataset with the options used in the version name,
                # to be saved in a txt file and used in the results plot.
                comment = "=" * LINE_LEN + "\n"
                v = f"v{v_str}: PRED={op_pred} + SPL={op_split} + BAL={op_bal} + SEM={op_sem}"

                comment += v + "\n" + "=" * LINE_LEN + "\n"
                comment += f"{'PRED'.ljust(15, '.')}: {OP_PREDICTORS_COMM[op_pred]}\n{'SPLIT'.ljust(15, '.')}: {OP_SLIT_COMM[op_split]}\n{'BALANCE'.ljust(15, '.')}: {OP_BALANCE_COMM[op_bal]:<20}\n{'MATCH SEM PREV'.ljust(15, '.')}: {OP_SEM_PREV_MATCH_COMM[op_sem]}"
                comment += '\n' + "-" * LINE_LEN + "\n"
                print(f"{comment}")

                # ====================================================================================
                line = "* BALANCE: "
                # ------------------------------------------------------------------------------------
                # Verify if the dataset is unbalanced and if it should be balanced based on the option selected.
                # Force the balance if automatic splip due to high level of unbalance in the original dataset
                # ====================================================================================
                comment += line
                print(line, end="")
                if op_bal == 'Yes' or op_split == '70_30':
                    # Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique)
                    features_for_balance = DS.drop(columns=['OC_Obs_Candid'])
                    target_for_balance = DS['OC_Obs_Candid']
                    smote = SMOTE(random_state=42)
                    X_balanced, y_balanced = smote.fit_resample(
                        features_for_balance, target_for_balance)
                    data_balanced = X_balanced.copy()
                    data_balanced['OC_Obs_Candid'] = y_balanced
                    line = f"Yes! Dataset balanced using SMOTE. From {Fun.count_classes(DS['OC_Obs_Candid'].values)} to {Fun.count_classes(y_balanced.values)}"
                    DS = data_balanced
                else:
                    line = f"No. Dataset remains unbalanced with {Fun.count_classes(DS['OC_Obs_Candid'].values)}"
                print(line)
                comment += line + "\n"

                # ====================================================================================
                line = "* PREDICTORS: "
                # ------------------------------------------------------------------------------------
                # Select the predictors based on the option selected.
                # ====================================================================================
                comment += line
                print(line, end="")
                if op_pred == 'ALL':
                    PREDICTORS = ['dia_prev', 'sem_prev', 'tsfc', 'magv', 'ur2m', 'Tx_Mod',
                                  'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod', 'Cl_Tx_Obs',
                                  'OC_Mod_Candid', 'IOC_Mod']
                elif op_pred == 'AD HOC':
                    PREDICTORS = ['sem_prev', 'tsfc', 'magv', 'ur2m', 'Tx_Mod',
                                  'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod', 'Cl_Tx_Obs']
                elif op_pred == 'POSIT_CORR':
                    PREDICTORS = ['Cl_Tx_Obs', 'P90cl_Tx_Obs', 'magv', 'tsfc']
                elif op_pred == 'POSIT_CORR+SEM_PREV':
                    PREDICTORS = ['Cl_Tx_Obs', 'P90cl_Tx_Obs',
                                  'magv', 'tsfc', 'sem_prev']

                line = f"{PREDICTORS}"
                print(line)
                comment += line + "\n"

                # ====================================================================================
                line = "* SPLITING THE FULL DATASET INTO TRAINING AND TESTING SETS: "
                # ------------------------------------------------------------------------------------
                # Verify the split option and split the dataset accordingly
                # ====================================================================================
                comment += line
                print(line, end="")
                if op_split == '11_19':
                    test_mask = (DS['data_prev'] >= 20231111) & (
                        DS['data_prev'] <= 20231119)
                    X_train_full = DS[~test_mask]
                    X_train_full.drop(columns=[TARGET], inplace=True)
                    y_train = DS[~test_mask][TARGET].values
                    X_test_full = DS[test_mask]
                    X_test_full.drop(columns=[TARGET], inplace=True)
                    y_test = DS[test_mask][TARGET].values
                    line += "Manual split, days 11-19 Nov 2023 for testing: "
                elif op_split == '70_30':
                    # Assure to do a balanced split in the datase in terms of OC_Obs_Candid, to avoid having too few samples of one class in the training or testing set
                    X = DS.drop(columns=[TARGET])
                    y = DS[TARGET]
                    X_train_full, X_test_full, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=X['OC_Obs_Candid'])
                    line += "Automatic split, 70% for training and 30% for testing: "

                line = f"{X_train_full.shape[0]} rows for training ({X_train_full.shape[0]/DS.shape[0]*100:.1f}%) and {X_test_full.shape[0]} rows for testing ({X_test_full.shape[0]/DS.shape[0]*100:.1f}%)."
                print(line)
                comment += line + "\n"

                line = f"  - The training set balance is now {Fun.count_classes(X_train_full['OC_Obs_Candid'])} and testing set is {Fun.count_classes(X_test_full['OC_Obs_Candid'])}."
                print(line)
                comment += line + "\n"

                # ====================================================================================
                line = "* SEM_PREV MATCH: "
                # ------------------------------------------------------------------------------------
                # Verify the SEM_PREV option and filter the training set accordingly
                # ====================================================================================
                comment += line
                print(line, end="")
                if op_sem == 'MatchTrainTest':
                    X_train_full = X_train_full[X_train_full['sem_prev'].isin(
                        X_test_full['sem_prev'].unique())]
                    y_train = y_train[X_train_full.index]
                    line = f"Yes! Training set filtered to match the sem_prev values in the test set. Now {X_train_full.shape[0]} rows for training and {X_test_full.shape[0]} rows for testing."
                else:
                    line = "No! Training set not filtered based on sem_prev values in the test set."

                print(line)
                comment += line + "\n"

                # ====================================================================================
                # From now on, common code for all versions, so we can save the results with the same version name and comment
                # ====================================================================================

                # Create x-axis labels combining data_rod and data_prev for the test set if split was '11-19'
                # otherwise use the index as labels
                if op_split == '11_19':
                    date_labels = X_test_full[[
                        'data_rod', 'data_prev']].astype(str)
                    date_labels = date_labels['data_rod'] + \
                        ' | ' + date_labels['data_prev']
                else:
                    date_labels = [x for x in range(1, len(X_test_full) + 1)]

                # Keep only the predictors in the X_train and X_test sets
                X_train = X_train_full[PREDICTORS]
                X_test = X_test_full[PREDICTORS]

                # Scaling (Crucial for Neural Networks)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # D. Tx_Mod as baseline model
                baseline_preds = X_test_full['Tx_Mod'].values

                # Collect models in a dictionary for easy iteration
                # Instantiate every iteration to avoid data leakage and ensure a fresh model for each version of the dataset.
                # df_eval = pd.DataFrame()
                models = {
                    "Eta": baseline_preds,
                    "RF": RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    ),
                    "XGB": xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        objective='reg:squarederror',
                        random_state=42
                    ),
                    "NN": MLPRegressor(
                        hidden_layer_sizes=(128, 64),  # (64, 32),
                        activation='relu',
                        solver='adam',
                        max_iter=10000,  # 1000
                        random_state=42
                    )
                }

                # ====================================================================================
                # Training and Evaluation
                # ====================================================================================
                line = "* TRAINING AND EVALUATION: "
                print(line)
                comment += line + "\n"

                line = "-" * LINE_LEN
                print(line)
                comment += line + "\n"

                pred_values = {}
                for name, model in models.items():
                    # 1) Train
                    # Note: Tree models (RF, XGB) don't strictly need scaling, but ANN does.
                    if name in ["RF", "XGB"]:
                        m = model.fit(X_train, y_train)  # No scaling
                    elif name in ["NN"]:
                        # Scaled data for ANN
                        m = model.fit(X_train_scaled, y_train)

                    # 2) Predict
                    preds = None
                    if name == "Eta":
                        preds = baseline_preds
                    elif name in ["RF", "XGB"]:
                        preds = model.predict(X_test)  # No scaling
                    else:
                        # Scaled data for ANN
                        preds = model.predict(X_test_scaled)

                    pred_values[name] = [round(float(x), 6) for x in preds]

                    # 3) Evaluate
                    mae = round(mean_absolute_error(y_test, preds), 4)
                    rmse = round(
                        float(np.sqrt(mean_squared_error(y_test, preds))), 4)
                    # Calculate correlation to Tx_Obs
                    corr = float(np.corrcoef(y_test, preds)[0, 1])

                    line = f"{name:>5} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | CORR: {corr:.4f}"
                    print(line)
                    comment += line + "\n"

                    # Append the predictions to the eval_df DataFrame for this version
                    df_eval = pd.concat([df_eval, pd.DataFrame({
                        "VERSION": v_str, "MODEL": name, "MAE": mae, "RMSE": rmse, "CORR": corr},
                        index=[0])], axis=0, ignore_index=False)

                    # Save a txt file with the version description
                    with open(f'{DIR_RESULTS_INFO}/INFO-v{v_str}.txt', 'w', encoding='utf-8') as f:
                        x = f.write(comment)

                # end of models loop

                line = "* Saving continuos calibrated predictions and results ... "
                print(line)
                comment += line + "\n"

                # Save predictions to a CSV file with the version name in the filename, to be used for the results plot at the end of all versions
                df_pred = X_test_full.copy()
                df_pred["Tx_Obs"] = y_test
                df_pred["Eta_Pred"] = pred_values["Eta"]
                df_pred["RF_Pred"] = pred_values["RF"]
                df_pred["XGB_Pred"] = pred_values["XGB"]
                df_pred["NN_Pred"] = pred_values["NN"]
                df_pred.to_csv(os.path.join(
                    DIR_RESULTS_PRED, f'PRED-v{v_str}.csv'), index=False)

                line = "* Saving binary classification predictions for binary analysis ..."
                print(line)
                comment += line + "\n"

                # Save one classification dataset per version with all model estimatives in columns
                df_pred_classif = df_pred.copy()
                for model_name in ["NN", "RF", "XGB"]:
                    pred_col = f"{model_name}_Pred"
                    classif_col = f"OC_{model_name}_Candid"
                    df_pred_classif[classif_col] = (
                        df_pred_classif[pred_col] >= df_pred_classif['Tx_Obs']
                    ).astype(int)

                df_pred_classif.to_csv(
                    os.path.join(
                        DIR_RESULTS_CLASSIF,
                        f'v{v_str}-PRED_CLASSIF.csv'
                    ),
                    index=False
                )

                line = "* Saving composite analysis plot ..."
                print(line)
                comment += line + "\n"

                fig = FPlot.plot_composite_analysis(
                    df_pred_classif=df_pred_classif,
                    models=models,
                    feature_names=PREDICTORS,
                    X_test=X_test,
                    X_test_scaled=X_test_scaled,
                    y_test=y_test,
                    v_str=v_str
                )
                fig.savefig(
                    os.path.join(
                        DIR_RESULTS_COMPOSITE,
                        f'RESULTS-v{v_str}-CompositeAnalysis.png'
                    ),
                    dpi=300
                )
                plt.close("all")

                line = "* Saving time series plots ..."
                print(line)
                comment += line + "\n"

                # ====================================================================================
                # Visualization
                # ====================================================================================

                # Option 1: Comprehensive dashboard (recommended)
                # fig = FPlot.plot_comprehensive_dashboard(
                #    y_test, X_test, pred_values, X_test_full, date_labels, v_str
                # )
                # fig.savefig(os.path.join(DIR_RESULTS, f'RESULTS-v{v_str}-Dashboard.png'), dpi=300)
                # if show_res:
                #    plt.show()
                # plt.close("all")

                # Option 2: Individual plot types (save all)
                # Time series with reduced ticks
                fig = FPlot.plot_time_series_reduced_ticks(
                    y_test, X_test_full, pred_values, date_labels, v_str
                )
                fig.savefig(os.path.join(
                    DIR_RESULTS_TIMESERIES, f'v{v_str}-TimeSeries.png'), dpi=300)
                plt.close("all")

                # Predicted vs Actual
                # fig = FPlot.plot_predicted_vs_actual(y_test, pred_values, v_str)
                # fig.savefig(os.path.join(DIR_RESULTS, f'RESULTS-v{v_str}-PredVsActual.png'), dpi=300)
                # plt.close("all")

                # Residuals
                # fig = FPlot.plot_residuals(y_test, pred_values, X_test_full, date_labels, v_str)
                # fig.savefig(os.path.join(DIR_RESULTS, f'RESULTS-v{v_str}-Residuals.png'), dpi=300)
                # plt.close("all")

                # Model subpanels
                fig = FPlot.plot_model_subpanels(
                    y_test, X_test_full, pred_values, date_labels, v_str
                )
                fig.savefig(os.path.join(
                    DIR_RESULTS_TIMESERIES, f'v{v_str}-Subpanels.png'), dpi=300)
                plt.close("all")

                v_idx += 1

            # end of SEM_PREV loop
        # end of BALANCE loop
    # end of SPLIT loop
# end of PREDICTORS loop


# Save df_eval with the version name in the filename, to be used for the results plot at the end of all versions
df_eval.to_csv(os.path.join(DIR_RESULTS, 'EVAL-METRICS.csv'), index=False)

# df_pred.columns
# df_pred[['data_rod', 'data_prev', 'dia_prev', 'sem_prev', 'Tx_Mod', 'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod', 'Cl_Tx_Obs',
#       'OC_Mod_Candid', 'IOC_Mod', 'Tx_Obs', 'Eta_Pred', 'RF_Pred', 'XGB_Pred', 'NN_Pred']]
