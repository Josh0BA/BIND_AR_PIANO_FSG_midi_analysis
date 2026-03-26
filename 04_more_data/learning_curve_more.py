import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from control_more import df_transitions

# =========================
# LEARNING CURVE PLOTTING
# =========================

BLOCK_ORDER = ["pretest", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "posttest"]


def map_test_to_block(test_name: str) -> str:
    """Normalisiert Test-Namen auf die Plot-Blocknamen."""
    test_lower = str(test_name).lower().strip()
    if 'pre' in test_lower:
        return 'pretest'
    if 'post' in test_lower:
        return 'posttest'
    if test_lower.startswith('b'):
        match = re.search(r'b\s*(\d+)', test_lower)
        if match:
            return f"b{match.group(1)}"
    if 'block' in test_lower:
        match = re.search(r'(\d+)', test_lower)
        if match:
            return f'b{match.group(1)}'
    return test_lower

def prepare_for_plotting(df_trans: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet Transition-Daten für das Plotting vor.
    Mappt Test-Namen zu Block-Namen, filtert Ausschlüsse und berechnet
    Mittelwerte pro Person, Block und transition_code.
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
    
    exclude_mask = pd.Series(False, index=df_plot.index)
    for pid, test in exclude_list:
        exclude_mask = exclude_mask | (
            (df_plot['Participant_ID'] == pid) & (df_plot['Test'] == test)
        )
    df_plot = df_plot.loc[~exclude_mask].copy()

    transition_time_column = 'transition_time' if 'transition_time' in df_plot.columns else 'onset_to_onset'

    df_plot['block'] = df_plot['Test'].apply(map_test_to_block)
    df_plot['freq'] = df_plot['frequency']
    df_plot['transition_time_s'] = df_plot[transition_time_column]

    df_plot = df_plot[df_plot['block'].isin(BLOCK_ORDER)].copy()

    df_plot = (
        df_plot.groupby(
            ['Participant_ID', 'Test', 'block', 'transition_code', 'freq'],
            observed=True,
            as_index=False
        )['transition_time_s']
        .mean()
    )

    df_plot['block'] = pd.Categorical(
        df_plot['block'],
        categories=BLOCK_ORDER,
        ordered=True
    )
    
    return df_plot

def compute_means(df: pd.DataFrame, freq_filter: str = None) -> pd.DataFrame:
    """
    Berechnet Mittelwerte und 95% Konfidenzintervalle der Übergangszeiten pro Block.
    Grundlage sind zuvor pro Person, Block und transition_code gemittelte Zeiten.
    
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
    )
    means.columns = ['block', 'mean_tt', 'ci_tt']
    return means


def compute_means_per_transition(df: pd.DataFrame, freq_filter: str = None) -> pd.DataFrame:
    """
    Berechnet Mittelwerte und 95% Konfidenzintervalle pro transition_code und Block.
    """
    if freq_filter:
        df = df[df['freq'] == freq_filter]

    def ci_95(x):
        if len(x) < 2:
            return 0
        return 1.96 * x.std() / np.sqrt(len(x))

    means = (
        df.groupby(['transition_code', 'block'], observed=True)['transition_time_s']
          .agg(['mean', ci_95])
          .reset_index()
    )
    means.columns = ['transition_code', 'block', 'mean_tt', 'ci_tt']
    means['transition_code'] = pd.to_numeric(means['transition_code'], errors='coerce')
    means = means.dropna(subset=['transition_code']).copy()
    means['transition_code'] = means['transition_code'].astype(int)
    return means


def _transition_label(transition_code: int) -> str:
    """Erzeugt eine lesbare Beschreibung aus einem Transition-Code (z.B. 12 -> 1->2)."""
    code_str = str(int(transition_code))
    if len(code_str) == 2:
        return f"{code_str} ({code_str[0]}->{code_str[1]})"
    return code_str

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

                participant_means = (
                    df_participant.groupby('block')['transition_time_s']
                      .mean()
                      .reset_index()
                      .rename(columns={'transition_time_s': 'mean_tt'})
                )

                training_participant = participant_means[participant_means['block'].isin(BLOCK_ORDER)].copy()
                training_participant['block'] = pd.Categorical(
                    training_participant['block'],
                    categories=BLOCK_ORDER,
                    ordered=True
                )
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
        
        means_df['block'] = pd.Categorical(means_df['block'], categories=BLOCK_ORDER, ordered=True)
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


def plot_all_transition_bars(df_all: pd.DataFrame, title: str, out_path: str) -> None:
    """
    Zeigt alle Transition-Datenpunkte als Balken in einem Plot.
    Farben kodieren h/s, x-Achse zeigt Transition-Beschreibungen.
    """
    plot_df = df_all.copy()
    plot_df['transition_code'] = pd.to_numeric(plot_df['transition_code'], errors='coerce')
    plot_df = plot_df.dropna(subset=['transition_code']).copy()
    plot_df['transition_code'] = plot_df['transition_code'].astype(int)

    if plot_df.empty:
        print(f"  Warnung: Keine Daten fuer {title} vorhanden")
        return

    plot_df = plot_df.sort_values(['transition_code', 'freq', 'block', 'Participant_ID']).reset_index(drop=True)

    # Positioniere Balken in Gruppen nach transition_code, damit alle Daten sichtbar bleiben.
    x_positions = np.zeros(len(plot_df), dtype=float)
    xtick_positions = []
    xtick_labels = []
    cursor = 0.0
    intra_step = 0.16
    group_gap = 1.0

    for transition_code in sorted(plot_df['transition_code'].unique()):
        mask = plot_df['transition_code'] == transition_code
        idxs = plot_df.index[mask].tolist()
        n = len(idxs)
        start = cursor
        for local_idx, data_idx in enumerate(idxs):
            x_positions[data_idx] = start + local_idx * intra_step
        end = start + max(0, (n - 1) * intra_step)
        xtick_positions.append((start + end) / 2)
        xtick_labels.append(_transition_label(transition_code))
        cursor = end + group_gap

    color_map = {'h': '#ff7f0e', 's': '#2ca02c'}
    bar_colors = [color_map.get(freq, '#7f7f7f') for freq in plot_df['freq']]

    plt.figure(figsize=(22, 9))
    plt.bar(
        x_positions,
        plot_df['transition_time_s'],
        color=bar_colors,
        width=0.13,
        alpha=0.75,
        linewidth=0
    )

    plt.title(title)
    plt.xlabel('Transition code (description)')
    plt.ylabel('Transition time (s)')
    plt.xticks(xtick_positions, xtick_labels, rotation=90)
    plt.grid(True, axis='y', alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label='frequent (h)', markerfacecolor=color_map['h'], markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='rare (s)', markerfacecolor=color_map['s'], markersize=10),
    ]
    plt.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_all_transition_curves_one_plot(means_df: pd.DataFrame, title: str, out_path: str) -> None:
    """
    Zeichnet alle Lernverlaeufe (pro transition_code) in EINEM Plot.
    Jede Transition erhaelt eine eigene Farbe; Transition 12 ist fix tuerkis.
    """
    plot_df = means_df.copy()
    plot_df['transition_code'] = pd.to_numeric(plot_df['transition_code'], errors='coerce')
    plot_df = plot_df.dropna(subset=['transition_code']).copy()
    plot_df['transition_code'] = plot_df['transition_code'].astype(int)

    if plot_df.empty:
        print(f"  Warnung: Keine Daten fuer {title} vorhanden")
        return

    transitions = sorted(plot_df['transition_code'].unique().tolist())
    x_positions = np.arange(len(BLOCK_ORDER))

    # Basispalette fuer alle Transitionen
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(transitions), 20)))
    color_map = {code: base_colors[idx % len(base_colors)] for idx, code in enumerate(transitions)}

    plt.figure(figsize=(16, 9))

    for transition_code in transitions:
        transition_data = plot_df[plot_df['transition_code'] == transition_code].copy()
        transition_data['block'] = pd.Categorical(transition_data['block'], categories=BLOCK_ORDER, ordered=True)
        transition_data = transition_data.sort_values('block')

        block_to_y = dict(zip(transition_data['block'].astype(str), transition_data['mean_tt']))
        y_values = [block_to_y.get(block, np.nan) for block in BLOCK_ORDER]

        plt.plot(
            x_positions,
            y_values,
            marker='o',
            linewidth=2,
            markersize=4,
            color=color_map[transition_code],
            alpha=0.95,
            label=_transition_label(transition_code)
        )

    plt.title(title)
    plt.xlabel('Block')
    plt.ylabel('Transition time (s)')
    plt.xticks(x_positions, BLOCK_ORDER)
    plt.grid(True, axis='y', alpha=0.25)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, title='Transitions')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_learning_curve_per_transition(means_df: pd.DataFrame, title: str, out_path: str, ncols: int = 3) -> None:
    """
    Small-multiples Plot: jedes Subplot zeigt einen transition_code separat.
    """
    transitions = sorted(means_df['transition_code'].unique().tolist())
    if not transitions:
        print(f"  Warnung: Keine Daten fuer {title} vorhanden")
        return

    nrows = int(np.ceil(len(transitions) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3.2), sharey=True)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    x_positions = np.arange(len(BLOCK_ORDER))

    for idx, transition_code in enumerate(transitions):
        ax = axes[idx]
        transition_data = means_df[means_df['transition_code'] == transition_code].copy()
        transition_data['block'] = pd.Categorical(transition_data['block'], categories=BLOCK_ORDER, ordered=True)
        transition_data = transition_data.sort_values('block')

        block_to_y = dict(zip(transition_data['block'].astype(str), transition_data['mean_tt']))
        block_to_ci = dict(zip(transition_data['block'].astype(str), transition_data['ci_tt']))

        y_values = [block_to_y.get(block, np.nan) for block in BLOCK_ORDER]
        ci_values = [block_to_ci.get(block, np.nan) for block in BLOCK_ORDER]

        ax.errorbar(
            x_positions,
            y_values,
            yerr=ci_values,
            linewidth=1.8,
            marker='o',
            markersize=4,
            capsize=3,
            color='#1f77b4',
            alpha=0.9,
            zorder=5
        )
        ax.set_title(f'Transition {transition_code}', fontsize=10)
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(BLOCK_ORDER, rotation=45, ha='right', fontsize=8)

    for idx in range(len(transitions), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.supxlabel('Block')
    fig.supylabel('Transition time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

# Erstelle Plot-Verzeichnis
PLOT_DIR = 'Plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Bereite Daten für Plotting vor
try:
    df_plot = prepare_for_plotting(df_transitions)
    
    # Berechne Mittelwerte für Gesamtkurven
    means_all = compute_means(df_plot)
    
    means_h = compute_means(df_plot, 'h')
    
    means_s = compute_means(df_plot, 's')
    
    # Erstelle Dictionary für Plot-Funktion
    means_dict = {
        'all': means_all[['block', 'mean_tt', 'ci_tt']],
        'frequent (h)': means_h[['block', 'mean_tt', 'ci_tt']],
        'rare (s)': means_s[['block', 'mean_tt', 'ci_tt']]
    }

    # Berechne Mittelwerte pro transition_code (separate Darstellung je Übergang)
    means_transition_all = compute_means_per_transition(df_plot)
    means_transition_h = compute_means_per_transition(df_plot, 'h')
    means_transition_s = compute_means_per_transition(df_plot, 's')
    
    # Neuer Hauptplot: alle Lernverlaeufe (pro Transition) in einem Plot
    plot_path = os.path.join(PLOT_DIR, 'learning_curve_all_more.png')
    plot_all_transition_curves_one_plot(
        means_transition_all,
        'All transition learning curves in one plot',
        plot_path
    )

    # Neue Darstellung: ein Teilplot pro transition_code
    plot_learning_curve_per_transition(
        means_transition_all,
        'Learning Curve per Transition Code (all)',
        os.path.join(PLOT_DIR, 'learning_curve_per_transition_all.png')
    )
    plot_learning_curve_per_transition(
        means_transition_h,
        'Learning Curve per Transition Code (frequent)',
        os.path.join(PLOT_DIR, 'learning_curve_per_transition_frequent.png')
    )
    plot_learning_curve_per_transition(
        means_transition_s,
        'Learning Curve per Transition Code (rare)',
        os.path.join(PLOT_DIR, 'learning_curve_per_transition_rare.png')
    )
    
    
    print("✅ Learning Curve Plot erfolgreich erstellt")
    
except Exception as e:
    print(f"⚠️  Fehler beim Erstellen der Plots: {e}")
