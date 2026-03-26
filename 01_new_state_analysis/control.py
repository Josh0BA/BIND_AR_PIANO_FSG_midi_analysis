import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Main import (
    df,
    aufwärmen,
    block,
    pre_post_test,
    map_transition_to_code,
    get_transition_frequency
)

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

# --- Export der Ergebnisse ---
# Hauptdaten (ohne die komplexe detected_states Spalte für CSV)
df_export = df.drop('detected_states', axis=1)
df_export.to_csv('sequence_analysis_results.csv', index=False, encoding='utf-8-sig')

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
    # Handle NaN or non-list values
    if not isinstance(states_list, list) or len(states_list) < 2:
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
