# Load Data -> MIDI -> Extract Pitches
import os
import re
from typing import Set, Dict, Tuple
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from midi_state_analysis.folder_utils import find_midi_data_folder

# ---------------------------------------------------------------------------
# 1. State-Definitionen (fixes Mapping deiner 9 Zustände)
# ---------------------------------------------------------------------------

STATE_DEFS: Dict[int, Set[int]] = {
    0: {65, 67, 72, 74, 77, 79},  # state1: F4, G4, C5, D5, F5, G5
    1: {60, 62, 64, 72, 77, 79},  # state2: C4, D4, E4, C5, F5, G5
    2: {60, 62, 64, 67, 72, 76},  # state3: C4, D4, E4, G4, C5, E5
    3: {60, 62, 64, 76, 77, 79},  # state4: C4, D4, E4, E5, F5, G5
    4: {62, 64, 65, 76, 77, 79},  # state5: D4, E4, F4, E5, F5, G5
    5: {64, 65, 72, 76, 77, 79},  # state6: E4, F4, C5, E5, F5, G5
    6: {60, 62, 72, 74, 76, 77},  # state7: C4, D4, C5, D5, E5, F5
    7: {64, 65, 72, 74, 77, 79},  # state8: E4, F4, C5, D5, F5, G5
    8: {62, 64, 65, 67, 72, 74},  # state9: D4, E4, F4, G4, C5, D5
}

aufwärmen = [
    6,
    3,
    0,
    8,
    4,
    8,
    3,
    6,
    4,
    1,
    7,
    1,
    0,
    5,
    8,
    5,
    3,
    1,
    4,
    3,
    2,
    0,
    4,
    0,
    3,
    7,
    6,
    1
]

block = [
    5,
    6,
    7,
    8,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    0,
    1,
    2,
    3,
    4,
    5,
    4,
    5,
    6,
    7,
    8,
    0,
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    7,
    8,
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    0,
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    0,
    1,
    2,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    0,
    1,
    2,
    3,
    4,
    5
]

pre_post_test = [
    8,
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8
]

# ---------------------------------------------------------------------------
# Übergangs-Mapping und Häufigkeiten (aus config.py)
# ---------------------------------------------------------------------------

# Mapping für jeden Übergang: Code -> Häufigkeit (h=häufig, s=selten)
TRANSITION_FREQUENCIES: Dict[int, str] = {
    12: "h", 13: "s", 23: "h", 24: "s", 32: "s", 34: "h", 45: "h", 46: "s", 
    56: "h", 57: "s", 65: "s", 67: "h", 78: "h", 79: "s", 81: "s", 89: "h", 
    91: "h", 98: "s"
}

def map_transition_to_code(from_state: int, to_state: int) -> int:
    """
    Mappt ein State-Paar auf den Übergangscode.
    
    Args:
        from_state: Ausgangs-State (0-8)
        to_state: Ziel-State (0-8)
    
    Returns:
        Übergangscode (z.B. 12 für 1->2)
    
    Beispiel:
        >>> map_transition_to_code(1, 2)
        12
        >>> map_transition_to_code(8, 0)
        80
    """
    # Da Main_short 0-basiert ist, config aber 1-basiert:
    # Wir addieren 1 zu jedem State
    from_display = from_state + 1
    to_display = to_state + 1
    
    return int(f"{from_display}{to_display}")

def get_transition_frequency(transition_code: int) -> str:
    """
    Gibt die Häufigkeit eines Übergangs zurück.
    
    Args:
        transition_code: Numerischer Code (z.B. 12, 89)
    
    Returns:
        'h' für häufig, 's' für selten, 'UNKNOWN' wenn nicht definiert
    """
    return TRANSITION_FREQUENCIES.get(transition_code, "UNKNOWN")

def get_expected_transitions(test_type: str) -> list:
    """
    Erstellt Liste der erwarteten Übergänge basierend auf der State-Sequenz.
    
    Args:
        test_type: Name des Tests (z.B. 'Pretest', 'B1', 'Aufwärmen')
    
    Returns:
        Liste von Übergangscodes in der erwarteten Reihenfolge
    """
    test_lower = test_type.lower()
    
    if 'auf' in test_lower:
        sequence = aufwärmen
    elif test_lower.startswith('b') or 'block' in test_lower:
        sequence = block
    elif 'pre' in test_lower or 'post' in test_lower:
        sequence = pre_post_test
    else:
        return []
    
    # Konvertiere State-Sequenz zu Übergangs-Sequenz
    transitions = []
    for i in range(len(sequence) - 1):
        code = map_transition_to_code(sequence[i], sequence[i + 1])
        transitions.append(code)
    
    return transitions

# Locate the MIDI data folder named "Daten (MIDI)" using folder_utils.
# If not found, fall back to the original hardcoded path.
root_folder = find_midi_data_folder(start_path='.')
if root_folder is None:
    # Plattformunabhängiger Fallback relativ zu dieser Datei
    root_folder = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Daten (MIDI)'))
    print(f"Warning: 'Daten (MIDI)' not found; falling back to {root_folder}")
else:
    print(f"Using MIDI data folder: {root_folder}")

data = []

# Walk through all subfolders
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        # Verarbeite alle MIDI-Dateien, Filterung nach Testtypen erfolgt später
        if filename.lower().endswith(('.mid', '.midi')):
            file_path = os.path.join(dirpath, filename)

            try:
                # Extract participant ID and test from filename
                # Format: MIDI_{ParticipantID}_{Test}.mid
                # Example: MIDI_BE16MI_Fingertest1.mid
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0].upper() == 'MIDI':
                    participant_id = parts[1]
                    test_name = parts[2].split('.')[0]  # Keep everything after second underscore

                # Extract note keystrokes and Timing

                # Load MIDI file
                midi = pretty_midi.PrettyMIDI(file_path)

                # - Store extracted pitches in an array named 'played_notes'
                # Extract played notes (non-drum, sorted by start time)
                played_notes = []
                for instrument in midi.instruments:
                    if not instrument.is_drum:
                        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
                        for note in sorted_notes:
                            name = pretty_midi.note_number_to_name(note.pitch)
                            played_notes.append({
                                'pitch': note.pitch,
                                'name': name,
                                'start': note.start,
                                'end': note.end
                            })
                
                # --- NEUE LOGIK: Sequenz-gesteuerte State-Erkennung ---
                
                # 1. Bestimme die Ziel-Sequenz basierend auf dem Test-Namen
                test_type = str(test_name).lower()
                if 'auf' in test_type:
                    target_sequence = aufwärmen
                elif test_type.startswith('b') or 'block' in test_type:
                    target_sequence = block
                elif 'pre' in test_type or 'post' in test_type:
                    target_sequence = pre_post_test
                else:
                    target_sequence = []

                detected_states = []
                current_note_idx = 0
                
                # --- SICHERERE LOGIK: Sequenz-gesteuert mit Abbruch-Limit ---
                max_search_range = 20  # Wie viele Noten darf das Skript "durchsuchen", um den State zu finden?
                
                # Wir gehen die Ziel-Zustände nacheinander durch
                for state_index, target_state_id in enumerate(target_sequence):
                    required_pitches = STATE_DEFS[target_state_id]
                    found_state = False
                    
                    # Wir begrenzen die Suche auf einen Bereich ab der aktuellen Note
                    search_limit = min(current_note_idx + max_search_range, len(played_notes))
                    
                    for i in range(current_note_idx, search_limit - 5):
                        # Kleines Fenster von 10 Noten für die Erkennung eines 6er-Akkords
                        window = played_notes[i : i + 10]
                        window_pitches = {n['pitch'] for n in window}
                        
                        if required_pitches.issubset(window_pitches):
                            # State gefunden! Extrahiere die relevanten Noten
                            seen_pitches = set()
                            state_notes = []
                            for n in window:
                                if n['pitch'] in required_pitches and n['pitch'] not in seen_pitches:
                                    state_notes.append(n)
                                    seen_pitches.add(n['pitch'])
                            
                            # Sortiere state_notes nach Zeit, um korrekte Start/End-Zeiten zu haben
                            state_notes = sorted(state_notes, key=lambda x: x['start'])
                            
                            detected_states.append({
                                'state': target_state_id,
                                'notes': state_notes,
                                'start_time': state_notes[0]['start'],
                                'end_time': state_notes[-1]['end'],
                                'target_index': state_index, # Position in der Soll-Sequenz
                                'notes_skipped': i - current_note_idx  # WICHTIG: Wie viele Noten waren "Müll" dazwischen?
                            })
                            
                            # Setze den Zeiger hinter die letzte Note dieses erkannten States
                            # damit wir nicht dieselben Noten für den nächsten State nutzen
                            last_note_in_state = max([played_notes.index(n) for n in state_notes])
                            current_note_idx = last_note_in_state + 1
                            found_state = True
                            break
                    
                    if not found_state:
                        # Hier markieren wir explizit eine Lücke in der Datenstruktur
                        detected_states.append({
                            'state': None, 
                            'expected_state': target_state_id,
                            'target_index': state_index,
                            'error': 'MISSING'
                        })
                        # Wir setzen current_note_idx NICHT weiter, um dem nächsten State 
                        # die Chance zu geben, ab der gleichen Stelle gefunden zu werden.
                        print(f"   ℹ️ State {target_state_id} an Position {state_index} nicht gefunden.")

                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Test': test_name, #example: Fingertest1, B2, Pretest
                    'detected_states': detected_states,
                    # 'all_pitches': [n['pitch'] for n in played_notes],  # Liste von Werten
                    # 'Keystrokes': len(played_notes),
                    # 'Correct_Sequences': correct_sequences,
                }

                data.append(info)

                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

#create DataFram
df = pd.DataFrame(data)
print(df)

# Falls keine Daten erkannt wurden, sauber beenden
if df.empty:
    print("Keine erkannten MIDI-Dateien oder keine passenden Ereignisse gefunden. Analyse wird beendet.")
    raise SystemExit(0)

#Control: Remove duplicate states, keeping only the last occurrence of consecutive identical states
def filter_consecutive_states(states_list):
    if not states_list:
        return []
    
    filtered = []
    for i in range(len(states_list)):
        current_state = states_list[i]['state']
        
        # Prüfen, ob dies das letzte Element ist ODER ob der nächste Zustand anders ist
        if i == len(states_list) - 1 or current_state != states_list[i + 1]['state']:
            filtered.append(states_list[i])
            
    return filtered

# Anwenden auf die Spalte im DataFrame
df['detected_states'] = df['detected_states'].apply(filter_consecutive_states)

# Kurze Kontrolle im Terminal
for idx, row in df.iterrows():
    states_only = [s['state'] for s in row['detected_states']]
    print(f"ID: {row['Participant_ID']} | Sequence: {states_only}")

# - Compare detected state sequence against expected sequence (from config)
# --- Analyse: Vergleich mit der Soll-Sequenz ---

def check_sequence_accuracy(row):
    """
    Compare detected state sequence against expected sequence based on test type.
    
    Returns a dictionary with accuracy metrics and error information.
    """
    # 1. Bestimme die Ziel-Sequenz basierend auf dem Test-Namen
    test_type = str(row['Test']).lower()
    
    if 'auf' in test_type:  # Deckt 'aufwärmen' ab
        target = aufwärmen
    elif test_type.startswith('b') or 'block' in test_type:
        target = block
    elif 'pre' in test_type or 'post' in test_type:
        target = pre_post_test
    else:
        target = []  # Default to empty instead of returning None
    
    # 2. Handle empty or missing detected_states
    detected_states = row.get('detected_states', [])
    if not detected_states:
        return {
            'accuracy_pct': 0.0,
            'correct_count': 0,
            'target_length': len(target),
            'actual_length': 0,
            'missing_states': len(target),
            'extra_states': 0,
            'error': 'No states detected'
        }
    
    if not target:
        return {
            'accuracy_pct': 0.0,
            'correct_count': 0,
            'target_length': 0,
            'actual_length': len(detected_states),
            'missing_states': 0,
            'extra_states': 0,
            'error': 'Unknown test type'
        }
    
    # 3. Extrahiere die tatsächlich gespielten Zustands-IDs
    actual = [s['state'] for s in detected_states]
    
    # 4. Vergleich - Position-basiert
    matches = 0
    for i in range(min(len(actual), len(target))):
        if actual[i] == target[i]:
            matches += 1
    
    # 5. Berechne Metriken
    accuracy = (matches / len(target)) * 100 if len(target) > 0 else 0
    missing = max(0, len(target) - len(actual))  # Zu wenige Zustände
    extra = max(0, len(actual) - len(target))     # Zu viele Zustände
    
    return {
        'accuracy_pct': round(accuracy, 2),
        'correct_count': matches,
        'target_length': len(target),
        'actual_length': len(actual),
        'missing_states': missing,
        'extra_states': extra
    }

# Anwenden auf den DataFrame
df['sequence_results'] = df.apply(check_sequence_accuracy, axis=1)

# Expandiere die Ergebnisse in separate Spalten für einfachere Analyse
df_results = pd.json_normalize(df['sequence_results'])
df = pd.concat([df.drop('sequence_results', axis=1), df_results], axis=1)

# --- Ausgabe der Ergebnisse ---
print("\n" + "="*100)
print("SEQUENZ-VERGLEICH: DETAILLIERTE ERGEBNISSE")
print("="*100)

for idx, row in df.iterrows():
    # Symbol basierend auf Genauigkeit
    if row['accuracy_pct'] == 100:
        symbol = "✅"
    elif row['accuracy_pct'] >= 80:
        symbol = "⚠️"
    else:
        symbol = "❌"
    
    print(f"\n{symbol} {row['Participant_ID']} - {row['Test']}")
    print(f"   Genauigkeit: {row['accuracy_pct']:.1f}% ({row['correct_count']}/{row['target_length']} korrekt)")
    print(f"   Länge: {row['actual_length']} erkannt vs. {row['target_length']} erwartet")
    
    if row.get('missing_states', 0) > 0:
        print(f"   ⚠️  Fehlende Zustände: {row['missing_states']}")
    if row.get('extra_states', 0) > 0:
        print(f"   ⚠️  Zusätzliche Zustände: {row['extra_states']}")
    if 'error' in row and pd.notna(row['error']):
        print(f"   ❌ Fehler: {row['error']}")
    
    # Zeige erkannte Sequenz
    detected_seq = [s['state'] for s in row['detected_states']]
    print(f"   Erkannt: {detected_seq}")

# --- Export der Ergebnisse ---
# Hauptdaten (ohne die komplexe detected_states Spalte für CSV)
df_export = df.drop('detected_states', axis=1)
df_export.to_csv('sequence_analysis_results.csv', index=False, encoding='utf-8-sig')
print(f"\n📊 Ergebnisse gespeichert in 'sequence_analysis_results.csv'")

# Analyze -> Compute transition times between states
# --- Analyse: Berechnung der Übergangszeiten (Transition Times) ---
def calculate_transitions(states_list, test_type: str = ""):
    """
    Berechnet die Zeitdifferenzen zwischen aufeinanderfolgenden Zuständen.
    Erweitert mit Übergangscodes und Häufigkeiten aus config.py.
    
    Args:
        states_list: Liste der erkannten States mit Timing-Informationen
        test_type: Name des Tests für erwartete Übergangs-Sequenz
    
    Returns:
        List of transition dictionaries with timing metrics and frequency info
    """
    if len(states_list) < 2:
        return []
    
    # Filter out states that are missing (have 'error': 'MISSING')
    valid_states = [s for s in states_list if 'start_time' in s and 'end_time' in s]
    
    if len(valid_states) < 2:
        return []
    
    transitions = []
    for i in range(len(valid_states) - 1):
        s1 = valid_states[i]
        s2 = valid_states[i + 1]
        
        # Berechnung der Intervalle
        onset_to_onset = s2['start_time'] - s1['start_time']
        offset_to_onset = s2['start_time'] - s1['end_time']  # "Die Lücke" (kann negativ sein bei Überlappung)
        
        # Zusätzliche Metriken
        state_duration = s1['end_time'] - s1['start_time']
        overlap = max(0, s1['end_time'] - s2['start_time'])  # Überlappung (falls negatives offset_to_onset)
        
        # NEUE FELDER: Übergangscode und Häufigkeit
        transition_code = map_transition_to_code(s1['state'], s2['state'])
        frequency = get_transition_frequency(transition_code)
        
        transitions.append({
            'from_state': s1['state'],
            'to_state': s2['state'],
            'transition_code': transition_code,  # z.B. 12, 89
            'frequency': frequency,  # 'h' oder 's'
            'onset_to_onset': round(onset_to_onset, 4),
            'offset_to_onset': round(offset_to_onset, 4),
            'state_duration': round(state_duration, 4),
            'overlap': round(overlap, 4),
            'position': i  # Position in der Sequenz
        })
    return transitions

# 1. Übergänge berechnen (mit Test-Typ für erwartete Sequenz)
df['transitions'] = df.apply(lambda row: calculate_transitions(row['detected_states'], row['Test']), axis=1)

# 2. Durchschnittliche Übergangszeit pro Versuch berechnen (als neue Metrik)
df['avg_transition_time'] = df['transitions'].apply(
    lambda x: round(np.mean([t['onset_to_onset'] for t in x]), 4) if x else np.nan
)

df['median_transition_time'] = df['transitions'].apply(
    lambda x: round(np.median([t['onset_to_onset'] for t in x]), 4) if x else np.nan
)

# 3. "Flüssigkeit" berechnen (Variabilität der Übergangszeiten)
# Ein niedrigerer Wert (CV) deutet auf einen gleichmäßigeren, gelernten Rhythmus hin.
def calculate_cv(trans):
    """Coefficient of Variation: std/mean"""
    if not trans:
        return np.nan
    times = [t['onset_to_onset'] for t in trans]
    if len(times) > 1 and np.mean(times) > 0:
        return round(np.std(times) / np.mean(times), 4)
    return np.nan

df['movement_variability_cv'] = df['transitions'].apply(calculate_cv)

# 4. Durchschnittliche Lücke zwischen Zuständen (Gap-Analyse)
df['avg_gap'] = df['transitions'].apply(
    lambda x: round(np.mean([t['offset_to_onset'] for t in x]), 4) if x else np.nan
)

# 5. Durchschnittliche Zustandsdauer
df['avg_state_duration'] = df['transitions'].apply(
    lambda x: round(np.mean([t['state_duration'] for t in x]), 4) if x else np.nan
)

# 6. Anzahl der Überlappungen (wo nächster Zustand beginnt, bevor vorheriger endet)
df['overlap_count'] = df['transitions'].apply(
    lambda x: sum(1 for t in x if t['overlap'] > 0) if x else 0
)

# 7. NEUE METRIKEN: Häufigkeits-basierte Analyse
df['frequent_transitions_avg_time'] = df['transitions'].apply(
    lambda x: round(np.mean([t['onset_to_onset'] for t in x if t.get('frequency') == 'h']), 4) if any(t.get('frequency') == 'h' for t in x) else np.nan
)

df['rare_transitions_avg_time'] = df['transitions'].apply(
    lambda x: round(np.mean([t['onset_to_onset'] for t in x if t.get('frequency') == 's']), 4) if any(t.get('frequency') == 's' for t in x) else np.nan
)
# --- Export der Ergebnisse: Übergangszeiten als flache Tabelle ---
transition_rows = []
for _, row in df.iterrows():
    pid = row['Participant_ID']
    test = row['Test']
    for t in row['transitions']:
        transition_rows.append({
            'Participant_ID': pid,
            'Test': test,
            'position': t['position'],
            'from_state': t['from_state'],
            'to_state': t['to_state'],
            'transition_code': t.get('transition_code', ''),
            'frequency': t.get('frequency', ''),
            'onset_to_onset': t['onset_to_onset'],
            'offset_to_onset': t['offset_to_onset'],
            'state_duration': t['state_duration'],
            'overlap': t['overlap'],
        })

df_transitions = pd.DataFrame(transition_rows)
df_transitions.to_csv('transition_times_results.csv', index=False, encoding='utf-8-sig')
print("\n📊 Übergangszeiten gespeichert in 'transition_times_results.csv'")

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
    print(f"📈 Plot gespeichert: {out_path}")

# Erstelle Plot-Verzeichnis
PLOT_DIR = 'Plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Bereite Daten für Plotting vor
print("\n" + "="*100)
print("LEARNING CURVE PLOTS")
print("="*100)

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

print("\n" + "="*100)
print("ANALYSE ABGESCHLOSSEN")
print("="*100)
