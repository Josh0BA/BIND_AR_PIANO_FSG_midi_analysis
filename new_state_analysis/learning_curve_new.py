import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from control import df_transitions

# =========================
# LEARNING CURVE PLOTTING
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
    block_order = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
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
    Berechnet Mittelwerte und 95% Konfidenzintervalle der Übergangszeiten pro Block.
    
    Args:
        df: DataFrame mit 'block', 'freq', 'transition_time_s' Spalten
        freq_filter: Optional 'h' oder 's' um nur häufige/seltene Übergänge zu nehmen
    """
    if freq_filter:
        df = df[df['freq'] == freq_filter]
    
    def ci_95(x):
        """Berechnet 95% Konfidenzintervall"""
        if len(x) < 2:
            return 0
        return 1.96 * x.std() / np.sqrt(len(x))
    
    means = (
        df.groupby('block', observed=True)['transition_time_s']
          .agg(['mean', ci_95])
          .reset_index()
          .rename(columns={'mean': 'mean_tt', '<lambda>': 'ci_tt'})
    )
    means.columns = ['block', 'mean_tt', 'ci_tt']
    return means

def plot_learning_curve_combined(means_dict: dict, title: str, out_path: str, df_original: pd.DataFrame = None) -> None:
    """
    Erstellt einen Plot mit häufigen und seltenen Übergängen.
    Mit Konfidenzintervallen und farbigen Participant-Linien.
    
    Args:
        means_dict: Dictionary {label: DataFrame mit 'block', 'mean_tt', 'ci_tt' Spalten}
        title: Plot-Titel
        out_path: Ausgabe-Pfad für PNG
        df_original: Original DataFrame mit allen Daten für Participant-Linien
    """
    plt.figure(figsize=(14, 6))
    
    # Farben für Participant-Linien (Palette mit vielen Farben)
    participant_colors = plt.cm.tab20c(np.linspace(0, 1, 20))
    
    # Plotte Participant-Linien mit Farben
    if df_original is not None:
        try:
            participants = sorted(df_original['Participant_ID'].unique())
            for idx, participant in enumerate(participants):
                df_participant = df_original[df_original['Participant_ID'] == participant].copy()
                
                # Mappe Test zu block mit pretest/posttest
                def map_test_to_block(test_name: str) -> str:
                    test_lower = str(test_name).lower().strip()
                    if 'pre' in test_lower:
                        return 'pretest'
                    elif 'post' in test_lower:
                        return 'posttest'
                    elif test_lower.startswith('b') and len(test_lower) <= 2:
                        return test_lower
                    elif 'block' in test_lower:
                        match = re.search(r'(\d+)', test_lower)
                        if match:
                            return f'b{match.group(1)}'
                    return test_lower
                
                df_participant['block'] = df_participant['Test'].apply(map_test_to_block)
                df_participant['transition_time_s'] = df_participant['onset_to_onset']
                
                # Gruppiere nach Block
                participant_means = (
                    df_participant.groupby('block')['transition_time_s']
                      .mean()
                      .reset_index()
                      .rename(columns={'transition_time_s': 'mean_tt'})
                )
                
                # Plot B1-B8 nur
                block_order = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
                training_participant = participant_means[participant_means['block'].isin(block_order)]
                training_participant['block'] = pd.Categorical(training_participant['block'], categories=block_order, ordered=True)
                training_participant = training_participant.sort_values('block')
                
                if not training_participant.empty:
                    plt.plot(
                        range(len(training_participant)),
                        training_participant['mean_tt'],
                        linewidth=0.8,
                        alpha=0.3,
                        color=participant_colors[idx % len(participant_colors)],
                        zorder=2
                    )
        except Exception as e:
            print(f"  Warnung: Participant-Linien konnten nicht geplottet werden: {e}")
    
    # Plotte die Durchschnittskurven mit CI für alle, häufig und selten
    color_map = {
        'all': '#1f77b4',
        'frequent (h)': '#ff7f0e',
        'rare (s)': '#2ca02c'
    }
    
    for label in ['all', 'frequent (h)', 'rare (s)']:
        if label not in means_dict:
            continue
        means_df = means_dict[label].sort_values('block')
        color = color_map.get(label, '#1f77b4')
        
        # Pretest + B1-B8 + Posttest
        block_order = ["pretest", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "posttest"]
        means_df['block'] = pd.Categorical(means_df['block'], categories=block_order, ordered=True)
        means_df = means_df.sort_values('block')
        
        # Plot mit Fehlerbalken (Konfidenzintervall)
        plt.errorbar(
            range(len(means_df)),
            means_df['mean_tt'],
            yerr=means_df['ci_tt'],
            linewidth=2.5,
            marker='o',
            markersize=7,
            color=color,
            label=label,
            capsize=5,
            capthick=1.5,
            alpha=0.8,
            zorder=10
        )
        
        # Setze x-Ticks und Labels
        plt.xticks(range(len(means_df)), means_df['block'].astype(str))
    
    plt.title(title)
    plt.xlabel('Block')
    plt.ylabel('Transition time (s)')
    plt.ylim(1.5, 2.6)
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
        'all': means_all[['block', 'mean_tt', 'ci_tt']],
        'frequent (h)': means_h[['block', 'mean_tt', 'ci_tt']],
        'rare (s)': means_s[['block', 'mean_tt', 'ci_tt']]
    }
    
    # Erstelle kombinierten Plot
    plot_path = os.path.join(PLOT_DIR, 'learning_curve_all_h_s.png')
    plot_learning_curve_combined(
        means_dict,
        'Learning curve during the acquisition phase',
        plot_path,
        df_original=df_plot  # Übergebe die gefilterten Daten für Participant-Linien
    )
    
    
    print("✅ Learning Curve Plot erfolgreich erstellt")
    
except Exception as e:
    print(f"⚠️  Fehler beim Erstellen der Plots: {e}")
