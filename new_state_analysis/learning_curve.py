import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from control import df_transitions

# =========================
# LEARNING CURVE PLOTTING
# =========================

def prepare_for_plotting(df_trans: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet transition_times_results.csv für die Plotting-Funktion vor.
    Mappt Test-Namen zu Block-Namen und benennt Spalten um.
    """
    df_plot = df_trans.copy()
    
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
    block_order = ["pretest", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "posttest"]
    df_plot = df_plot[df_plot['block'].isin(block_order)]
    
    # Setze Categorical für korrekte Sortierung
    df_plot['block'] = pd.Categorical(
        df_plot['block'],
        categories=block_order,
        ordered=True
    )
    
    return df_plot

def compute_means(df: pd.DataFrame, freq_filter: str = None) -> pd.DataFrame:
    """
    Berechnet Mittelwerte der Übergangszeiten pro Block.
    
    Args:
        df: DataFrame mit 'block', 'freq', 'transition_time_s' Spalten
        freq_filter: Optional 'h' oder 's' um nur häufige/seltene Übergänge zu nehmen
    """
    if freq_filter:
        df = df[df['freq'] == freq_filter]
    
    means = (
        df.groupby('block', observed=True)['transition_time_s']
          .mean()
          .reset_index()
          .rename(columns={'transition_time_s': 'mean_tt'})
    )
    return means

def plot_learning_curve_combined(means_dict: dict, title: str, out_path: str) -> None:
    """
    Erstellt einen kombinierten Plot mit mehreren Kurven.
    Pretest und Posttest werden als separate Punkte ohne Verbindung zu B1-B8 dargestellt.
    
    Args:
        means_dict: Dictionary {label: DataFrame mit 'block' und 'mean_tt' Spalten}
        title: Plot-Titel
        out_path: Ausgabe-Pfad für PNG
    """
    plt.figure(figsize=(12, 5))
    
    for label, means_df in means_dict.items():
        means_df = means_df.sort_values('block')
        
        # Teile die Daten in drei Segmente: pretest, b1-b8, posttest
        pretest_data = means_df[means_df['block'] == 'pretest']
        training_data = means_df[means_df['block'].isin(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'])]
        posttest_data = means_df[means_df['block'] == 'posttest']
        
        # Plot pretest (einzelner Punkt, keine Linie)
        if not pretest_data.empty:
            plt.plot(
                pretest_data['block'].astype(str),
                pretest_data['mean_tt'],
                marker='o',
                markersize=8,
                linestyle='None',
                label=label if training_data.empty and posttest_data.empty else None
            )
        
        # Plot training blocks (b1-b8) mit verbundener Linie
        if not training_data.empty:
            plt.plot(
                training_data['block'].astype(str),
                training_data['mean_tt'],
                linewidth=2,
                marker='o',
                label=label
            )
        
        # Plot posttest (einzelner Punkt, keine Linie)
        if not posttest_data.empty:
            plt.plot(
                posttest_data['block'].astype(str),
                posttest_data['mean_tt'],
                marker='o',
                markersize=8,
                linestyle='None',
                label=None  # Keine separate Label für posttest
            )
    
    plt.title(title)
    plt.xlabel('Block')
    plt.ylabel('Transition time (s)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# Erstelle Plot-Verzeichnis
PLOT_DIR = 'Plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Bereite Daten für Plotting vor
try:
    df_plot = prepare_for_plotting(df_transitions)
    
    # Berechne Mittelwerte für alle, häufige und seltene Übergänge
    means_all = compute_means(df_plot)
    means_all['label'] = 'all'
    
    means_h = compute_means(df_plot, 'h')
    means_h['label'] = 'frequent (h)'
    
    means_s = compute_means(df_plot, 's')
    means_s['label'] = 'rare (s)'
    
    # Erstelle Dictionary für Plot-Funktion
    means_dict = {
        'all': means_all[['block', 'mean_tt']],
        'frequent (h)': means_h[['block', 'mean_tt']],
        'rare (s)': means_s[['block', 'mean_tt']]
    }
    
    # Erstelle kombinierten Plot
    plot_path = os.path.join(PLOT_DIR, 'learning_curve_all_h_s.png')
    plot_learning_curve_combined(
        means_dict,
        'Learning curve during the acquisition phase',
        plot_path
    )
    
    
    print("✅ Learning Curve Plot erfolgreich erstellt")
    
except Exception as e:
    print(f"⚠️  Fehler beim Erstellen der Plots: {e}")
