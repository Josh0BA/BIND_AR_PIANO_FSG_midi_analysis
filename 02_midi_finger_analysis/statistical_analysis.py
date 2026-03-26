import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import zscore
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt  
import math

import os
from midi_state_analysis.folder_utils import find_midi_data_folder

# Find the MIDI data folder and look for fingergeschicklichkeit.csv in the parent directory
midi_folder = find_midi_data_folder(start_path='.')
if midi_folder:
    csv_path = os.path.join(os.path.dirname(midi_folder), "fingergeschicklichkeit.csv")
else:
    # Fallback to old path if MIDI folder not found
    csv_path = "src/data_analysis_pipeline/Data/fingergeschicklichkeit.csv"

df_finger = pd.read_csv(csv_path)

# --------- fuctions -----------
def analyze_scores(df):
    df_numeric = df.drop(columns=['Participant_ID'], errors='ignore')

    # Sample size and degrees of freedom
    n = len(df_numeric)
    dfree = n - 1

    # Confidence level
    confidence = 95
    alpha = (100 - confidence) / 100  # e.g. 0.05 for 95% CI

    # Calculate stats
    mean = df_numeric.mean()
    std = df_numeric.std()  # Standard Deviation
    se = std / np.sqrt(n)  # Standard Error

    # TINV equivalent: two-tailed t critical value
    t_critical = t.ppf(1 - alpha / 2, dfree)

    # Confidence interval bounds
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se

    # Combine into summary table
    df_summary = pd.DataFrame({
        'Mean': mean,
        'Std Dev': std,
        'Std Error': se,
        'CI Lower (95%)': ci_lower,
        'CI Upper (95%)': ci_upper,
        'T Critical': t_critical
    })

    df_numeric = df.select_dtypes(include=np.number)
    for col in df_numeric.columns:
        stat, p = shapiro(df_numeric[col])
        normality = p # if p < 0.05, the data is not normally distributed
        df_summary.loc[col, 'Normality'] = normality
   
    return df_summary

def prepare_pretest_posttest_data(df):
    """
    Reshapes wide format data to long format for Pretest vs Posttest comparison.
    Returns a dataframe with columns: Participant_ID, Timepoint, Measure, Value
    """
    # Select only correct columns for analysis
    pretest_cols = [col for col in df.columns if 'Fingertest1_correct' in col or 'Fingertest2_correct' in col]
    posttest_cols = [col for col in df.columns if 'Fingertest3_correct' in col or 'Fingertest4_correct' in col]
    
    data_long = []
    for idx, row in df.iterrows():
        participant_id = row['Participant_ID']
        
        # Pretest values
        for col in pretest_cols:
            if not pd.isna(row[col]):
                data_long.append({
                    'Participant_ID': participant_id,
                    'Timepoint': 'Pretest',
                    'Measure': col,
                    'Value': row[col]
                })
        
        # Posttest values
        for col in posttest_cols:
            if not pd.isna(row[col]):
                data_long.append({
                    'Participant_ID': participant_id,
                    'Timepoint': 'Posttest',
                    'Measure': col,
                    'Value': row[col]
                })
    
    return pd.DataFrame(data_long)

def analyze_by_timepoint(df):
    """
    Performs separate analysis for Pretest and Posttest groups.
    Returns summaries for both groups.
    """
    # Get average scores for each participant
    pretest_cols = [col for col in df.columns if 'Fingertest1_correct' in col or 'Fingertest2_correct' in col]
    posttest_cols = [col for col in df.columns if 'Fingertest3_correct' in col or 'Fingertest4_correct' in col]
    
    df_analysis = pd.DataFrame()
    df_analysis['Participant_ID'] = df['Participant_ID']
    df_analysis['Pretest_avg'] = df[pretest_cols].mean(axis=1)
    df_analysis['Posttest_avg'] = df[posttest_cols].mean(axis=1)
    
    # Create separate dataframes for analysis
    df_pretest = pd.DataFrame({
        'Participant_ID': df['Participant_ID'],
        'Score': df_analysis['Pretest_avg']
    })
    
    df_posttest = pd.DataFrame({
        'Participant_ID': df['Participant_ID'],
        'Score': df_analysis['Posttest_avg']
    })
    
    summary_pretest = analyze_scores(df_pretest)
    summary_posttest = analyze_scores(df_posttest)
    
    return summary_pretest, summary_posttest, df_pretest, df_posttest

def compare_groups_ttest(summary1, summary2, data1, data2, group1_name="Group1", group2_name="Group2"):
    """
    Performs independent t-test or Mann-Whitney U test between two groups.
    """
    # Sample sizes
    n1 = len(data1)
    n2 = len(data2)
    
    results = []
    
    for col in summary1.index:
        if col not in summary2.index:
            continue
            
        # Get data
        group1 = data1[col].dropna() if col in data1.columns else data1['Score'].dropna()
        group2 = data2[col].dropna() if col in data2.columns else data2['Score'].dropna()
        
        # Get normality p-values
        p_norm1 = summary1.loc[col, 'Normality']
        p_norm2 = summary2.loc[col, 'Normality']
        
        # Means and standard deviations
        mean1 = summary1.loc[col, 'Mean']
        mean2 = summary2.loc[col, 'Mean']
        sd1 = summary1.loc[col, 'Std Dev']
        sd2 = summary2.loc[col, 'Std Dev']
        
        # Decide test based on normality
        if p_norm1 >= 0.05 and p_norm2 >= 0.05:
            # Welch's t-test
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
            
            # Cohen's d
            pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
            cohen_d = (mean1 - mean2) / pooled_sd if pooled_sd != 0 else np.nan
            
            # Effect size label
            if np.isnan(cohen_d):
                effect_label = "N/A"
            elif abs(cohen_d) < 0.2:
                effect_label = "Negligible"
            elif abs(cohen_d) < 0.5:
                effect_label = "Small"
            elif abs(cohen_d) < 0.8:
                effect_label = "Medium"
            else:
                effect_label = "Large"
        else:
            # Mann–Whitney U test
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U"
            cohen_d = None
            effect_label = "N/A"
        
        results.append({
            'Measure': col,
            f'Mean {group1_name}': mean1,
            f'Mean {group2_name}': mean2,
            'Test': test_name,
            'p-value': p_value,
            'Significant (p < 0.05)': p_value < 0.05,
            "Cohen's d": cohen_d,
            'Effect Size': effect_label
        })
    
    return pd.DataFrame(results)

def remove_outliers_z(df): # apply to all numeric columns (excluding ID and Category)
    df_numeric = df.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(df_numeric))
    mask = (z_scores < 3).all(axis=1)  # keep only rows where all values are within 3 std
    return df[mask]

def check_normality_shapiro(df, group_name=""):
    df_numeric = df.select_dtypes(include=np.number)
    for col in df_numeric.columns:
        stat, p = shapiro(df_numeric[col])
        print(f"{group_name} - {col}: W={stat:.3f}, p={p:.3f} {'(Not normal)' if p < 0.05 else '(Normal)'}")

def grubbs_test(values, alpha=0.05):
    """
    Applies one-sided Grubbs' test for a single outlier.
    Returns the index of the outlier, or None if no outlier is found.
    """
    n = len(values)
    if n < 3:
        return None

    mean_y = np.mean(values)
    std_y = np.std(values, ddof=1)
    abs_diffs = np.abs(values - mean_y)
    max_dev_idx = abs_diffs.idxmax()
    G_calculated = abs_diffs[max_dev_idx] / std_y

    # Critical value from Grubbs' distribution
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    if G_calculated > G_critical:
        return max_dev_idx
    else:
        return None

def replace_outliers_with_nan(raw_df, summary_df, alpha=0.05, verbose=False):
    """
    Replaces outlier values with NaN in raw_df, based on normality info from summary_df.
    Uses Grubbs' test for normal variables, IQR method otherwise.
    
    Parameters:
    - raw_df: original data with numeric columns and 'Participant_ID'
    - summary_df: DataFrame with index = variable names and a 'Normality' column
    - alpha: significance level for Grubbs test (default 0.05)
    - verbose: if True, prints which values were set to NaN
    """
    df_clean = raw_df.copy()
    df_clean.set_index("Participant_ID", inplace=True)
    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col not in summary_df.index:
            continue
        
        values = df_clean[col].dropna()
        if len(values) < 3:
            continue  # Not enough data to test
        
        p_normal = summary_df.loc[col, "Normality"]
        is_normal = p_normal > 0.05

        if is_normal:
            # Grubbs test
            outlier_idx = grubbs_test(values, alpha=alpha)
            if outlier_idx is not None:
                df_clean.loc[outlier_idx, col] = np.nan
                if verbose:
                    print(f"[Grubbs] Outlier in '{col}' at participant {outlier_idx} set to NaN")
        else:
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (values < lower) | (values > upper)
            for idx in values.index[outlier_mask]:
                df_clean.loc[idx, col] = np.nan
                if verbose:
                    print(f"[IQR] Outlier in '{col}' at participant {idx} set to NaN")

    df_clean.reset_index(inplace=True)
    return df_clean

# ----------- prepare data and clean --------------

# JE13CL is a special case, played wrong sequence therefore just take the his played sequence calculated in saving_JE13CL.py
Fingertest1_correct = 16
Fingertest2_correct = 26

df_finger.loc[df_finger['Participant_ID'] == 'JE13CL', 'Fingertest1_correct'] = Fingertest1_correct
df_finger.loc[df_finger['Participant_ID'] == 'JE13CL', 'Fingertest2_correct'] = Fingertest2_correct



# calculate the mean, standard deviation, standard error and confidence interval
df_finger_summary = analyze_scores(df_finger)

print("\n=== Overall Summary ===")
print(df_finger_summary)

# Analyze Pretest vs Posttest
print("\n=== Pretest vs Posttest Analysis ===")
summary_pretest, summary_posttest, df_pretest, df_posttest = analyze_by_timepoint(df_finger)

print("\nPretest Summary:")
print(summary_pretest)

print("\nPosttest Summary:")
print(summary_posttest)

# Compare Pretest vs Posttest
comparison_results = compare_groups_ttest(summary_pretest, summary_posttest, df_pretest, df_posttest, "Pretest", "Posttest")
print("\nPretest vs Posttest Comparison:")
print(comparison_results)

# remove outliers using Grubbs test and IQR method
df_finger_clean = replace_outliers_with_nan(df_finger, df_finger_summary, alpha=0.05, verbose=True)


# -------- print the results -----------
# print(df_finger_summary)




