"""Regression for IOC_Mod calibration using Tx_Obs."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

v = "70-30"  # 70-30 | 11-19
show_res = True     # Show on screen
LINE_LEN = 55
DIR_RESULTS = 'results/regression'
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

# Load the DS
DS = pd.read_csv("1.2-DS_BALANCED.csv")
print(DS.columns)

# Split the dataset into features and target variable
X = DS[['data_prev', 'dia_prev', 'sem_prev', 'tsfc', 'magv', 'ur2m',
       'Tx_Mod', 'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod',
        'Cl_Tx_Obs', 'OC_Candid', 'IOC_Mod']]
y = DS['Tx_Obs']


if v == "70-30":
    print("Mode 70-30: 70% for training and 30% for testing with IOC_Obs equiparation")
    print("            Train and Test sets balanced in terms of IOC_Obs")
    # Split the dataset into training and testing sets in a manner that IOC_Obs distribution is preserved
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        DS.index.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=DS['IOC_Obs']
    )
if v == "11-19":
    print("Mode 11-19: Days 11 to 19 of November 2023 used for testing and the rest for training")
    print("            Test set totally inbalanced due to contains only IOC_Obs = 1")
    # Split the dataset into training and testing sets in a manner that testing lies
    # between 20231111 and 20131119
    test_mask = (DS['data_prev'] >= 20231111) & (DS['data_prev'] <= 20231119)
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Remove the data_prev column from both training and testing sets
    X_train.drop(columns=['data_prev'], inplace=True)
    X_test.drop(columns=['data_prev'], inplace=True)


# Mode 11-9: 520, 202
# Mode 70-30: 577, 145
print(len(X_train), len(X_test))


class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float


# Scaling (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. Model Initialization
# ==========================================

# A. Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# B. XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)

# C. Artificial Neural Network (MLP)
# Using 2 hidden layers with 64 and 32 neurons, ReLU activation
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# D. Tx_Mod as baseline model
baseline_preds = X_test['Tx_Mod'].values

# Collect models in a dictionary for easy iteration
models = {
    "Eta": baseline_preds,
    "RF": rf_model,
    "XGB": xgb_model,
    "NN": ann_model
}

# ==========================================
# 4. Training and Evaluation
# ==========================================

# Baseline: How bad was the original physical model?
raw_mae = mean_absolute_error(y_test, X_test['Tx_Mod'])
raw_rmse = np.sqrt(mean_squared_error(y_test, X_test['Tx_Mod']))
print(f"BASELINE (Raw Model) -> MAE: {raw_mae:.4f} | RMSE: {raw_rmse:.4f}\n")

results = {}

for name, model in models.items():
    # Train
    # Note: Tree models (RF, XGB) don't strictly need scaling, but ANN does.

    if name in ["RF", "XGB"]:
        model.fit(X_train, y_train)  # No scaling
    elif name in ["NN"]:
        model.fit(X_train_scaled, y_train)  # Scaled data for ANN

    # Predict
    if name == "Eta":
        preds = baseline_preds
    elif name in ["RF", "XGB"]:
        preds = model.predict(X_test)  # No scaling
    else:
        preds = model.predict(X_test_scaled)  # Scaled data for ANN

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = preds
    print(f"{name:<15} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")


# ==========================================
# 5. Visualization
# ==========================================
plt.figure(figsize=(14, 6))
results2 = results.copy()
results2.pop("Eta", None)  # Remove Eta from results to avoid plotting it again
# list(results.keys())

# Plot a subset of the test data for clarity (first 20 days)
n_view = 20
x_ax = range(n_view)

plt.plot(x_ax, y_test.values[:n_view], label='Observed (Truth)',
         color='black', linewidth=2, linestyle='-')
plt.plot(x_ax, X_test['Tx_Mod'].values[:n_view],
         label='Eta Model (Uncalibrated)', color='gray', linestyle='--')

colors = ['green', 'blue', 'red']

for (name, preds), color in zip(results2.items(), colors):
    plt.plot(x_ax, preds[:n_view], label=f'{name}', color=color, alpha=0.7)

plt.title(f'Model Calibration Comparison (First {n_view} Test Days)')
plt.ylabel('Max Temperature')
plt.xlabel('Days')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(DIR_RESULTS, f'00-RESULTS-Regression_Comparison_{v}.png'), dpi=300)
if show_res:
    plt.show()
plt.close("all")

# Save predictions to CSV
pred_df = pd.DataFrame({
    'Tx_Obs': y_test,
    'Tx_Mod': X_test['Tx_Mod'],
    'RF_Pred': results['RF'],
    'XGB_Pred': results['XGB'],
    'NN_Pred': results['NN']
})
pred_df.to_csv(
    f'{DIR_RESULTS}/00-RESULTS-Regression_Comparison_{v}.csv', index=False)


# Find out with forecast (Eta, RF, XGB ou NN) is more correlated to Tx_Obs
correlations = {name: np.corrcoef(y_test, preds)[0, 1]
                for name, preds in results.items()}
print("\nCorrelation with Observed (Tx_Obs):")
for name, corr in correlations.items():
    print(f"{name:<15} -> Correlation: {corr:.4f}")
# Save correlations to CSV
corr_df = pd.DataFrame.from_dict(
    correlations, orient='index', columns=['Correlation'])
corr_df.to_csv(os.path.join(
    DIR_RESULTS, f'00-RESULTS-Regression_Correlation_{v}.csv'))
