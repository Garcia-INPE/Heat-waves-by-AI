"""
Analyses the dataset, checks for missing values, shows summary statistics,
and creates visualizations to understand the distribution of the data features.
It also applies SMOTE to balance the dataset for classification tasks.

*** For classification purposes the model can't use IOC_Mod nor IOC_Obs as 
    features because they are set to 1 only if there is at least 3 consecutive occurrences
    
"""

from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make pandas display all columns
pd.set_option('display.max_columns', 0)

# Load the dataset
# data = pd.read_csv('input_final4.csv', header=0)
data = pd.read_csv('input_final4.csv', header=0, sep=';', decimal=',')
print(data.dtypes)
print(data.columns)

# Adjust the column names
data.columns = [
    'data_rod', 'data_prev', 'dia_prev', 'sem_prev', 'tsfc', 'magv', 'ur2m',
    'Tx_Mod', 'Tx_Obs', 'P90cl_Tx_Mod', 'P90cl_Tx_Obs', 'Cl_Tx_Mod',
    'Cl_Tx_Obs', 'IOC_Mod', 'IOC_Obs'
]
print(data)

# Summary statistics
print(data.describe())

# Round Tx_Mod, Tx_Obs, P90cl_Tx_Mod, P90cl_Tx_Obs to 5 decimal places
# Because they Tx_Mod and Tx_Obs have 5 decimal places and they are compared with P90cl_Tx_Mod and P90cl_Tx_Obs
# so they should have the same number of decimal places for better comparison and to avoid issues with floating point precision
data['Tx_Mod'] = data['Tx_Mod'].round(5)
data['Tx_Obs'] = data['Tx_Obs'].round(5)
data['P90cl_Tx_Mod'] = data['P90cl_Tx_Mod'].round(5)
data['P90cl_Tx_Obs'] = data['P90cl_Tx_Obs'].round(5)

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

# Check for missing values
print(data.isnull().sum())

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
data.to_csv('1.1-DATA_ORIGINAL-FULL.csv', index=False)

# Split the dataset into testing (data_prev between 20231111 and 20231119)
# and training (the rest of data_prev)
test_mask = (data['data_prev'] >= 20231111) & (
    data['data_prev'] <= 20231119)
data_train = data[~test_mask][['data_prev', 'Tx_Mod', 'P90cl_Tx_Mod', 'Tx_Obs',
                               'P90cl_Tx_Obs', 'OC_Mod_Candid', "IOC_Mod", 'OC_Obs_Candid', "IOC_Obs"]]
data_test = data[test_mask][['data_prev', 'Tx_Mod', 'P90cl_Tx_Mod', 'Tx_Obs',
                             'P90cl_Tx_Obs', 'OC_Mod_Candid', "IOC_Mod", 'OC_Obs_Candid', "IOC_Obs"]]
print(len(data_train), len(data_test))

# Fix the unbalanced train dataset using SMOTE (synthetic minority oversampling)
features_for_balance = data_train.drop(columns=['IOC_Obs_'])
target_for_balance = data_train[['IOC_Obs']]
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(
    features_for_balance, target_for_balance)

data_balanced = X_balanced.copy()
data_balanced['IOC_Obs'] = y_balanced
print(data_balanced['IOC_Obs'].value_counts())

# Save balanced dataset to CSV
data_balanced.to_csv('1.4-DATA_TRAIN-BALANCED.csv', index=False)

# EDA Visualizations

# Show the distribution of numerical features in a single plot
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
data_features = data.select_dtypes(include=['float64']).columns

# Show the histograms for data features in a single plot, use the adequate number of bins to show the distribution properly
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

# Show the box plots for data features in a single plot, avoiding showing two canvas.
# Make one subplot for magv and put it in top left, one for ur2m and put it on top right,
# and one for the rest of features and put it at the bottom,
# use shared y-axis in this sub-plot for better comparison
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
