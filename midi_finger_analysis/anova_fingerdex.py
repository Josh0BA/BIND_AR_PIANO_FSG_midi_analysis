import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import zscore
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt  
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
from scipy.stats import ttest_rel

# Get CSV path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
csv_path = os.path.join(script_dir, "..", "fingergeschicklichkeit.csv")

df_finger = pd.read_csv(csv_path)



# LN15CH is a special case, played wrong sequence therefore just take the his played sequence calculated in saving_JE13CL.py
Finger_1_correct = 15
Finger_2_correct = 16
Finger_3_correct = 16
Finger_4_correct = 16

df_finger.loc[df_finger['Participant_ID'] == 'LN15CH', 'Fingerttest_1_correct'] = Finger_1_correct
df_finger.loc[df_finger['Participant_ID'] == 'LN15CH', 'Fingerttest_2_correct'] = Finger_2_correct
df_finger.loc[df_finger['Participant_ID'] == 'LN15CH', 'Fingerttest_3_correct'] = Finger_3_correct
df_finger.loc[df_finger['Participant_ID'] == 'LN15CH', 'Fingerttest_4_correct'] = Finger_4_correct

## Anovatest
df_finger_clean = df_finger.rename(columns=lambda x: x.replace('-', '_'))

# List of columns to test
cols_to_test = [col for col in df_finger_clean.columns if ('correct' in col or 'keys' in col)]

# print(cols_to_test)

# Participants with any zero in the test metrics
zero_mask = (df_finger_clean[cols_to_test] == 0).any(axis=1)
df_sus_data = df_finger_clean[zero_mask]
print(f"\nParticipants with any value == 0: {len(df_sus_data)}")
if not df_sus_data.empty:
    to_show = ['Participant_ID'] if 'Participant_ID' in df_sus_data.columns else []
    print(df_sus_data[to_show + cols_to_test])


for col in cols_to_test:
    # print(f"\n--- Column: {col} ---")
    
    # Descriptive statistics
    # print(f"Mean: {df_finger_clean[col].mean():.2f}")
    # print(f"Std: {df_finger_clean[col].std():.2f}")
    # print(f"Min: {df_finger_clean[col].min():.2f}")
    # print(f"Max: {df_finger_clean[col].max():.2f}")
    
    # Normality test
    stat, p = shapiro(df_finger_clean[col])
    # print(f"Shapiro-Wilk: p = {p:.4f} {'✓ Normal' if p > 0.05 else '✗ Not normal'}")


# --------- plot the results -----------
# Select columns to plot
cols = [c for c in df_finger_clean.columns if 'correct' in c or 'keys' in c]
num_plots = len(cols)

# Set grid size
cols_per_row = 4  # Adjust this to control layout
rows = math.ceil(num_plots / cols_per_row)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(6 * cols_per_row, 5 * rows))
axes = axes.flatten()  # Flatten in case of multi-row layout-

# Plot each boxplot
for i, col in enumerate(cols):
    ax = axes[i]
    sns.boxplot(y=col, data=df_finger_clean, ax=ax)
    sns.stripplot(y=col, data=df_finger_clean, color='black', size=6, jitter=True, ax=ax)
    ax.set_title(f'{col}', fontsize=16)
    ax.set_ylabel('Value', fontsize=14)

    ax.tick_params(axis='y', labelsize=12)

    # Set y-axis limits
    if 'correct' in col:
        ax.set_ylim(-2, 35)
    elif 'keys' in col:
        ax.set_ylim(-10, 180)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()