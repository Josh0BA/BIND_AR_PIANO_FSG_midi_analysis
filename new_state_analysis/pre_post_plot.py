import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from control import df_transitions

# =========================
# Pre-/ Posttest Box PLOTTING
# =========================

def prepare_for_plotting(df_trans: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet transition_times_results.csv für die Plotting-Funktion vor.
    Mappt Test-Namen zu Block-Namen und benennt Spalten um.
    Filtert ausgeschlossene Participant-Test-Kombinationen.
    """
    df_plot = df_trans.copy()
    
    # Definiere Ausschluss-Liste (Participant_ID, Test)
    exclude_list = [
        ('FR01AN', 'B1'),
        ('FR01AN', 'B2'),
        ('KA17NA', 'B7'),
        ('KA17NA', 'Posttest'),
        ('OB09DA', 'B3'),
        ('TH31AL', 'B1'),
        ('TH31AL', 'Pretest'),
        ('ZUI21RA', 'B7'),
        ('ZUI21RA', 'B8'),
    ]
    
    # Filtere Ausschluss-Kombinationen
    for pid, test in exclude_list:
        df_plot = df_plot[~((df_plot['Participant_ID'] == pid) & (df_plot['Test'] == test))]
    
    # Mappe Test-Namen zu Block-Namen
    def map_test_to_block(test_name: str) -> str:
        test_lower = str(test_name).lower().strip()
        if 'pre' in test_lower:
            return 'pretest'
        elif 'post' in test_lower:
            return 'posttest'
        elif test_lower.startswith('b') and len(test_lower) <= 2:
            # B1, B2, etc.
            return test_lower
        elif 'block' in test_lower:
            # Extrahiere Nummer wenn vorhanden
            match = re.search(r'(\d+)', test_lower)
            if match:
                return f'b{match.group(1)}'
        return test_lower  # Fallback
    
    df_plot['block'] = df_plot['Test'].apply(map_test_to_block)
    df_plot['freq'] = df_plot['frequency']
    df_plot['transition_time_s'] = df_plot['onset_to_onset']
    
    # Filtere nur relevante Blöcke
    test_order = ["pretest", "posttest"]
    df_plot = df_plot[df_plot['block'].isin(test_order)]
    
    # Setze Categorical für korrekte Sortierung
    df_plot['block'] = pd.Categorical(
        df_plot['block'],
        categories=test_order,
        ordered=True
    )
    
    return df_plot


def plot_pre_post_boxplots(df_plot: pd.DataFrame, out_dir: str = 'Plots') -> None:
    """
    Erstellt Boxplots für Pre- und Posttest mit Stripplots.
    
    Args:
        df_plot: DataFrame mit Pre/Post-Test Daten
        out_dir: Ausgabe-Verzeichnis für Plots
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Metriken zum Plotten
    metrics = ['onset_to_onset']
    metric_labels = ['Transition time']
    
    # Erstelle Boxplots für die Metriken
    num_plots = len(metrics)
    cols_per_row = 1
    rows = num_plots
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        sns.boxplot(
            data=df_plot,
            x='block',
            y=metric,
            hue='freq',
            ax=ax,
            palette='Set2'
        )
        sns.stripplot(
            data=df_plot,
            x='block',
            y=metric,
            hue='freq',
            color='black',
            size=2,
            alpha=0.4,
            jitter=True,
            ax=ax,
            dodge=True,
            legend=False
        )
        
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Test', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='Frequency', loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'pre_post_boxplots.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅ Pre-/Post-Test Boxplot erstellt: {plot_path}")


def plot_pre_post_violin(df_plot: pd.DataFrame, out_dir: str = 'Plots') -> None:
    """
    Erstellt Violin Plots für Pre- und Posttest mit Stripplots.
    
    Args:
        df_plot: DataFrame mit Pre/Post-Test Daten
        out_dir: Ausgabe-Verzeichnis für Plots
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Metriken zum Plotten
    metrics = ['onset_to_onset']
    metric_labels = ['Transition time']
    
    # Erstelle Violin Plots für die Metriken
    num_plots = len(metrics)
    cols_per_row = 1
    rows = num_plots
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        sns.violinplot(
            data=df_plot,
            x='block',
            y=metric,
            hue='freq',
            ax=ax,
            palette='Set2'
        )
        sns.stripplot(
            data=df_plot,
            x='block',
            y=metric,
            hue='freq',
            color='black',
            size=2,
            alpha=0.4,
            jitter=True,
            ax=ax,
            dodge=True,
            legend=False
        )
        
        ax.set_title(f'{label} (Violin Plot)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Test', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='Frequency', loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'pre_post_violin.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅ Pre-/Post-Test Violin Plot erstellt: {plot_path}")


# Hauptausführung
if __name__ == "__main__":
    try:
        df_plot = prepare_for_plotting(df_transitions)
        plot_pre_post_boxplots(df_plot)
        plot_pre_post_violin(df_plot)
        print("✅ Pre-/Post-Test Analyse abgeschlossen")
    except Exception as e:
        print(f"⚠️  Fehler: {e}")
