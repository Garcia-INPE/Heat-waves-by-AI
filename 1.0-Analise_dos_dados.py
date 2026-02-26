"""
Exploratory Data Analysis (EDA)

Analyses the dataset, checks for missing values, shows summary statistics,
and creates visualizations to understand the distribution of the data features.
It also applies SMOTE to balance the dataset for classification tasks.

*** For classification purposes the model can't use IOC_Mod nor IOC_Obs as 
    features because they are set to 1 only if there is at least 3 consecutive occurrences
   
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make pandas display all columns
pd.set_option('display.max_columns', 0)

DIR_RESULTS = 'output/1.0-analise_dos_dados'
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

# Load the dataset
# data = pd.read_csv('input_final5.csv', header=0)
data = pd.read_csv('input/input_final4.csv', header=0, sep=';', decimal=',')

print("Missing values per column:")
print(data.isnull().sum())

# Adjust the column names
print(data.dtypes)
print(data.columns)
data.columns = [
    'data_rod', 'data_prev', 'dia_prev', 'sem_prev', 'tsfc', 'magv', 'ur2m',
    'Tx_Mod', 'Tx_Obs', 'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod',
    'Cl_Tx_Obs', 'IOC_Mod', 'IOC_Obs'
]
print(data.columns)

# Summary statistics
print(data.describe())

# Round all float columns to 6 decimal places due to comparisons
float_cols = data.select_dtypes(include=['float64']).columns
data[float_cols] = data[float_cols].round(6)

# Recalculate dia_prev and sem_prev columns
data.drop(columns=['dia_prev'], errors='ignore', inplace=True)
data.drop(columns=['sem_prev'], errors='ignore', inplace=True)
# dia_prev must be the difference in days between data_rod and data_prev
# Insert dia_prev column after data_prev
data.insert(data.columns.get_loc('data_prev')+1, 'dia_prev', (pd.to_datetime(
    data['data_prev'], format='%Y%m%d') - pd.to_datetime(data['data_rod'], format='%Y%m%d')).dt.days)
# sem_prev must be ceil(dia_prev/7)
# Insert sem_prev column after dia_prev
data.insert(data.columns.get_loc('dia_prev')+1, 'sem_prev', (data['dia_prev'] /
                                                             7).apply(lambda x: int(x) + 1 if x % 1 > 0 else int(x)))

# print the dataset sorted by dia_prev
print(data.sort_values(by='dia_prev'))

# Check the distribution of the dia_prev column, sort it by dia_prev values and show as horizontal table
print(data['dia_prev'].value_counts().sort_index().to_frame().T)
print(data['sem_prev'].value_counts().sort_index().to_frame().T)

# Count and remove rows where dia_prev is greater than 60
print(f"Rows before filtering: {len(data)}")
data = data[data['dia_prev'] <= 60]
print(f"Rows after filtering: {len(data)}")

# List rows where IOC_Obs equals 1
print(data[data['IOC_Obs'] == 1])

# Include a new column before IOC_Mod that indicates if Tx_Mod is greater than or equal to P90cl_Tx_Mod
# and convert it to integer (1 if true, 0 if false)
data.drop(columns=['OC_Mod_Candid'], errors='ignore', inplace=True)
data.insert(data.columns.get_loc('IOC_Mod'), 'OC_Mod_Candid',
            (data['Tx_Mod'] >= data['P90cl_Tx_Mod']).astype(int))
# Include a new column before IOC_Obs that indicates if Tx_Obs is greater than or equal to P90cl_Tx_Obs
# and convert it to integer (1 if true, 0 if false)
data.drop(columns=['OC_Obs_Candid'], errors='ignore', inplace=True)
data.insert(data.columns.get_loc('IOC_Obs'), 'OC_Obs_Candid',
            (data['Tx_Obs'] >= data['P90cl_Tx_Obs']).astype(int))

# Using the column IOC_Obs, check if the dataset is balanced: 361 x 22
print(data['OC_Obs_Candid'].value_counts())
print(data['OC_Mod_Candid'].value_counts())

# Sort data by data_rod and data_prev
data.sort_values(by=['data_rod', 'data_prev'], inplace=True)

# Save the "ORIGINAL' dataset to CSV
data.to_csv(os.path.join(
    DIR_RESULTS, '1.1-DATA_ORIGINAL-FULL.csv'), index=False)

# Visualizations

# Show the distribution of numerical features in a single plot
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
data_features = data.select_dtypes(include=['float64']).columns

# ===================================================================
# Show the histograms for data features in a single plot
# ===================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(data_features):
    # Use Freedman-Diaconis rule for optimal bin width
    q75, q25 = data[feature].quantile([0.75, 0.25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data[feature]) ** (1/3))
    n_bins = int((data[feature].max() - data[feature].min()
                  ) / bin_width) if bin_width > 0 else 30
    n_bins = min(max(n_bins, 10), 50)  # Keep bins between 10 and 50

    axes[idx].hist(data[feature], bins=n_bins,
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(feature, fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Value', fontsize=8)
    axes[idx].set_ylabel('Frequency', fontsize=8)
    axes[idx].grid(alpha=0.3)

plt.suptitle('Distribution of Data Features (Optimized Bins)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ===================================================================
# Box plots for data features in a single plot
# ===================================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[
    1, 1.2], hspace=0.3, wspace=0.2)

# Top left: magv
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(y=data['magv'], ax=ax1,
            color='lightcoral', width=0.4)
ax1.set_title('Box Plot of magv',
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# Top right: ur2m
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(y=data['ur2m'], ax=ax2,
            color='lightseagreen', width=0.4)
ax2.set_title('Box Plot of ur2m',
              fontsize=12, fontweight='bold')
ax2.set_ylabel('Value', fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Bottom: rest of features with shared y-axis
ax3 = fig.add_subplot(gs[1, :])
other_features = [
    f for f in data_features if f not in ['magv', 'ur2m']]
data_melted = data[other_features].melt(
    var_name='Feature', value_name='Value')
sns.boxplot(data=data_melted, x='Feature',
            y='Value', ax=ax3, palette='Set2')
ax3.set_title('Box plots of temp-related data features (Shared Y-axis)',
              fontsize=12, fontweight='bold')
ax3.set_ylabel('Value', fontsize=10)
ax3.set_xlabel('Feature', fontsize=10)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3, axis='y')

plt.suptitle('Box Plot Analysis - Data Features',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Pairplot for data features = SPLOM
sns.pairplot(data[data_features])
plt.suptitle('Pairplot of data features', y=1.02)
plt.show()
