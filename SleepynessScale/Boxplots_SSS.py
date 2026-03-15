import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import t

# 1. Daten laden
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, 'SleepinessScale_TN.xlsx')
csv_path = os.path.join(script_dir, 'SleepinessScale_TN.csv')

if os.path.exists(excel_path):
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except Exception as excel_error:
        try:
            df = pd.read_excel(excel_path, engine='calamine')
            print("Hinweis: XLSX via 'calamine' geladen (openpyxl konnte die Datei nicht lesen).")
        except Exception as calamine_error:
            if os.path.exists(csv_path):
                print("Hinweis: XLSX konnte nicht gelesen werden. Nutze stattdessen CSV.")
                df = pd.read_csv(csv_path)
            else:
                raise ValueError(
                    "XLSX konnte weder mit openpyxl noch mit calamine gelesen werden. "
                    f"openpyxl: {excel_error}; calamine: {calamine_error}. "
                    f"Bitte als CSV exportieren nach: {csv_path}"
                )
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(
       f"Keine Datei gefunden. Erwartet: {excel_path} oder {csv_path}"
    )

# 2. Die relevanten Spalten auswählen und ins Long-Format bringen
data_to_plot = df[['Pretest', 'Lernphase', 'Posttest']].melt(
    var_name='Test-Phasen',
    value_name='Skala-Werte'
)
phase_order = ['Pretest', 'Lernphase', 'Posttest']
data_to_plot['Test-Phasen'] = pd.Categorical(
    data_to_plot['Test-Phasen'], categories=phase_order, ordered=True
)

# 3. Deskriptive Statistik berechnen und als CSV speichern
summary = data_to_plot.groupby('Test-Phasen', observed=False)['Skala-Werte'].agg(
    mean='mean',
    std='std',
    median='median',
    min='min',
    max='max',
    count='count'
)
summary['se'] = summary['std'] / np.sqrt(summary['count'])
summary['t_critical'] = summary['count'].apply(lambda n: t.ppf(0.975, n - 1) if n > 1 else np.nan)
summary['ci95'] = summary['t_critical'] * summary['se']
summary['ci95_lower'] = summary['mean'] - summary['ci95']
summary['ci95_upper'] = summary['mean'] + summary['ci95']

summary_export = summary.reset_index().rename(
    columns={
        'Test-Phasen': 'Phase',
        'mean': 'Mittelwert',
        'std': 'Standardabweichung',
        'median': 'Median',
        'min': 'Minimum',
        'max': 'Maximum',
        'count': 'N',
        'ci95': '95%_KI_Halbbreite',
        'ci95_lower': '95%_KI_unten',
        'ci95_upper': '95%_KI_oben'
    }
)
stats_csv_path = os.path.join(script_dir, 'SleepinessScale_deskriptive_statistik.csv')
summary_export.to_csv(stats_csv_path, index=False)
print(f"Deskriptive Statistik gespeichert: {stats_csv_path}")

# 4. Grafik-Style festlegen
sns.set_style("whitegrid")

# 5a. Grafik 1: Boxplot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    x='Test-Phasen',
    y='Skala-Werte',
    data=data_to_plot,
    order=phase_order,
    color='lightblue'
)
sns.stripplot(
    x='Test-Phasen',
    y='Skala-Werte',
    data=data_to_plot,
    order=phase_order,
    color='black',
    size=5,
    alpha=0.7,
    jitter=True,
    ax=ax
)

# 6a. Beschriftung Grafik 1
plt.title('Sleepiness Scale: Boxplots', fontsize=16)
plt.ylabel('Skala-Werte', fontsize=14)
plt.xlabel('Test-Phasen', fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(0.5, 7)
ax.set_yticks(np.arange(1, 8, 1))
plt.tight_layout()

# 5b. Grafik 2: Mittelwerte mit 95%-Konfidenzintervallen
plt.figure(figsize=(10, 6))
ax = sns.stripplot(
    x='Test-Phasen',
    y='Skala-Werte',
    data=data_to_plot,
    order=phase_order,
    color='black',
    size=4,
    alpha=0.35,
    jitter=True,
    ax=ax
)

x_pos = np.arange(len(phase_order))
means = summary.loc[phase_order, 'mean'].to_numpy()
ci95 = summary.loc[phase_order, 'ci95'].to_numpy()

ax.errorbar(
    x=x_pos,
    y=means,
    yerr=ci95,
    fmt='D-',
    color='#1f77b4',
    linewidth=2,
    capsize=6,
    markersize=8,
    zorder=3
)

# 6b. Beschriftung Grafik 2
plt.title('Sleepiness Scale: Mittelwerte mit 95%-Konfidenzintervallen', fontsize=16)
plt.ylabel('Skala-Werte', fontsize=14)
plt.xlabel('Test-Phasen', fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.35)
sns.despine(ax=ax)

# 6. Anzeigen
plt.tight_layout()
plt.show()