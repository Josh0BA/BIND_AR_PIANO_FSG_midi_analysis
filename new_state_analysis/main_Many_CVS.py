# Load Data -> MIDI -> Extract Pitches
import os
import re
from typing import Set, Dict
import pretty_midi
import pandas as pd
import numpy as np

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

# Locate the MIDI data folder named "Daten (MIDI)" using folder_utils.
# If not found, fall back to the original hardcoded path.
root_folder = find_midi_data_folder(start_path='.')
if root_folder is None:
    root_folder = r"..\Daten (MIDI)"
    print(f"Warning: 'Daten (MIDI)' not found; falling back to {root_folder}")
else:
    print(f"Using MIDI data folder: {root_folder}")

data = []

# Walk through all subfolders
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.mid', '.midi')) and ('B' in filename.upper() or 'P' in filename.upper()):
            file_path = os.path.join(dirpath, filename)

            try:
                # Extract participant ID and test/attempt from filename
                # New format: MIDI_{ParticipantID}_{TestAndAttempt}.mid
                # Example: MIDI_BE16MI_Fingertest1.mid (test name + attempt suffix)
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0].upper() == 'MIDI':
                    participant_id = parts[1]
                    test_attempt = parts[2].split('.')[0]

                # Split test name and attempt (attempt expected as trailing digits)
                match = re.match(r"([A-Za-z]+)(\d+)", test_attempt)
                if match:
                    test_name = match.group(1)
                    attempt = match.group(2)
                else:
                    test_name = test_attempt
                    attempt = '1'

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
                # - Analyze pitches in 10-note windows (iterate through array in steps of 10)
                detected_states = []
                window_size = 10

                #to do am schluss noch anpassen dass dann halt window nur noch 9,8,7,6 weniger als 6 macht keinen sinn da stat min 6
                for i in range(0, len(played_notes) - window_size + 1, window_size): #+1??
                    window_notes = played_notes[i:i + window_size]
                    window_pitches = {note['pitch'] for note in window_notes}
                    
                    # Find matching state: check if all 6 expected state pitches are in the window
                    # (additional pitches are ignored)
                    best_match = None
                    best_match_pitches = None
                    
                    for state_num, state_pitches in STATE_DEFS.items():
                        # Subset matching: all state pitches must be in window pitches
                        if state_pitches.issubset(window_pitches):
                            best_match = state_num
                            best_match_pitches = state_pitches
                            break
                    
                    # - When a state is recognized, save the state with associated pitches, start time, and end time
                    if best_match is not None:
                        # Pick at most one note per expected pitch, taking the earliest occurrence
                        seen_pitches = set()
                        state_notes = []
                        for note in window_notes:
                            if note['pitch'] in best_match_pitches and note['pitch'] not in seen_pitches:
                                state_notes.append(note)
                                seen_pitches.add(note['pitch'])
                        detected_states.append({
                            'state': best_match,
                            'notes': state_notes,
                            'start_time': window_notes[0]['start'],
                            'end_time': window_notes[-1]['end'],
                            'window_index': i // window_size 
                        })

                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Test': test_name, #example: Fingertest
                    'Attempt': attempt, #example: 2
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
    elif 'block' in test_type:
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
    
    print(f"\n{symbol} {row['Participant_ID']} - {row['Test']} (Versuch {row['Attempt']})")
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

# --- Zusammenfassung nach Test-Typ ---
print("\n" + "="*100)
print("ZUSAMMENFASSUNG NACH TEST-TYP")
print("="*100)

summary = df.groupby('Test').agg({
    'accuracy_pct': ['mean', 'std', 'min', 'max'],
    'correct_count': 'mean',
    'Participant_ID': 'count'
}).round(2)

summary.columns = ['Genauigkeit_Mittel_%', 'Genauigkeit_Std_%', 'Genauigkeit_Min_%', 
                   'Genauigkeit_Max_%', 'Korrekte_Zustände_Mittel', 'Anzahl_Versuche']
print(summary)

# --- Zusammenfassung nach Teilnehmer ---
print("\n" + "="*100)
print("ZUSAMMENFASSUNG NACH TEILNEHMER")
print("="*100)

participant_summary = df.groupby('Participant_ID').agg({
    'accuracy_pct': 'mean',
    'correct_count': 'sum',
    'Test': 'count'
}).round(2)

participant_summary.columns = ['Durchschnittliche_Genauigkeit_%', 'Gesamt_Korrekte_Zustände', 'Anzahl_Tests']
participant_summary = participant_summary.sort_values('Durchschnittliche_Genauigkeit_%', ascending=False)
print(participant_summary)

# --- Export der Ergebnisse ---
# Hauptdaten (ohne die komplexe detected_states Spalte für CSV)
df_export = df.drop('detected_states', axis=1)
df_export.to_csv('sequence_analysis_results.csv', index=False, encoding='utf-8-sig')
print(f"\n📊 Ergebnisse gespeichert in 'sequence_analysis_results.csv'")

# Zusammenfassungen
summary.to_csv('summary_by_test.csv', encoding='utf-8-sig')
participant_summary.to_csv('summary_by_participant.csv', encoding='utf-8-sig')
print(f"📊 Zusammenfassungen gespeichert in 'summary_by_test.csv' und 'summary_by_participant.csv'")

print("\n" + "="*100)
print("ANALYSE ABGESCHLOSSEN")
print("="*100)

# Analyze -> Compute transition times between states
# --- Analyse: Berechnung der Übergangszeiten (Transition Times) ---
def calculate_transitions(states_list):
    """
    Berechnet die Zeitdifferenzen zwischen aufeinanderfolgenden Zuständen.
    
    Returns:
        List of transition dictionaries with timing metrics
    """
    if len(states_list) < 2:
        return []
    
    transitions = []
    for i in range(len(states_list) - 1):
        s1 = states_list[i]
        s2 = states_list[i + 1]
        
        # Berechnung der Intervalle
        onset_to_onset = s2['start_time'] - s1['start_time']
        offset_to_onset = s2['start_time'] - s1['end_time']  # "Die Lücke" (kann negativ sein bei Überlappung)
        
        # Zusätzliche Metriken
        state_duration = s1['end_time'] - s1['start_time']
        overlap = max(0, s1['end_time'] - s2['start_time'])  # Überlappung (falls negatives offset_to_onset)
        
        transitions.append({
            'from_state': s1['state'],
            'to_state': s2['state'],
            'transition_name': f"{s1['state']}->{s2['state']}",
            'onset_to_onset': round(onset_to_onset, 4),
            'offset_to_onset': round(offset_to_onset, 4),
            'state_duration': round(state_duration, 4),
            'overlap': round(overlap, 4),
            'position': i  # Position in der Sequenz
        })
    return transitions

# 1. Übergänge berechnen
df['transitions'] = df['detected_states'].apply(calculate_transitions)

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

# --- Erweiterte Ausgabe ---
print("\n" + "="*100)
print("ZEITANALYSE: ÜBERGÄNGE UND RHYTHMUS")
print("="*100)

for idx, row in df.iterrows():
    print(f"\n{'='*80}")
    print(f"ID: {row['Participant_ID']} - {row['Test']} (Versuch {row['Attempt']})")
    print(f"{'='*80}")
    
    if not row['transitions']:
        print("  ⚠️  Keine Übergänge gefunden (weniger als 2 Zustände erkannt)")
        continue
    
    print(f"  📊 Übergangszeiten (Onset-to-Onset):")
    print(f"     Mittelwert: {row['avg_transition_time']:.3f} sek")
    print(f"     Median:     {row['median_transition_time']:.3f} sek")
    print(f"     Variabilität (CV): {row['movement_variability_cv']:.3f}")
    
    print(f"\n  ⏱️  Timing-Details:")
    print(f"     Durchschn. Lücke (Gap): {row['avg_gap']:.3f} sek")
    print(f"     Durchschn. Zustandsdauer: {row['avg_state_duration']:.3f} sek")
    print(f"     Überlappungen: {row['overlap_count']} von {len(row['transitions'])} Übergängen")
    
    # Optional: Zeige die ersten 5 Übergänge als Beispiel
    if row['transitions']:
        print(f"\n  🔍 Erste Übergänge:")
        for i, t in enumerate(row['transitions'][:5], 1):
            overlap_marker = " (Überlappung!)" if t['overlap'] > 0 else ""
            print(f"     {i}. {t['transition_name']:8s}: {t['onset_to_onset']:6.3f}s | Gap: {t['offset_to_onset']:6.3f}s{overlap_marker}")
        if len(row['transitions']) > 5:
            print(f"     ... und {len(row['transitions']) - 5} weitere")

# --- Zusammenfassung nach Test-Typ ---
print("\n" + "="*100)
print("ZUSAMMENFASSUNG: ZEITMETRIKEN NACH TEST-TYP")
print("="*100)

time_summary = df.groupby('Test').agg({
    'avg_transition_time': ['mean', 'std'],
    'median_transition_time': ['mean'],
    'movement_variability_cv': ['mean', 'std'],
    'avg_gap': ['mean'],
    'avg_state_duration': ['mean'],
    'overlap_count': ['sum', 'mean']
}).round(4)

time_summary.columns = ['_'.join(col).strip() for col in time_summary.columns.values]
print(time_summary)

# --- Zusammenfassung nach Teilnehmer ---
print("\n" + "="*100)
print("ZUSAMMENFASSUNG: ZEITMETRIKEN NACH TEILNEHMER")
print("="*100)

participant_time_summary = df.groupby('Participant_ID').agg({
    'avg_transition_time': 'mean',
    'movement_variability_cv': 'mean',
    'overlap_count': 'sum'
}).round(4)

participant_time_summary.columns = ['Durchschn_Übergangszeit', 'Durchschn_Variabilität_CV', 'Gesamt_Überlappungen']
participant_time_summary = participant_time_summary.sort_values('Durchschn_Übergangszeit')
print(participant_time_summary)

# --- Export für vertiefte Statistik (JASP/SPSS/R) ---
print("\n" + "="*100)
print("DATENEXPORT")
print("="*100)

# 1. Erstellt eine flache Liste aller Übergänge für alle Probanden (Long-Format)
all_transitions = []
for idx, row in df.iterrows():
    for t in row['transitions']:
        t_data = {
            'Participant_ID': row['Participant_ID'],
            'Test': row['Test'],
            'Attempt': row['Attempt'],
            'Accuracy_Pct': row.get('accuracy_pct', np.nan),
            **t
        }
        all_transitions.append(t_data)

df_trans_long = pd.DataFrame(all_transitions)
if not df_trans_long.empty:
    df_trans_long.to_csv('detailed_transitions.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Detaillierte Übergangszeiten gespeichert: 'detailed_transitions.csv' ({len(df_trans_long)} Übergänge)")
else:
    print("⚠️  Keine Übergänge zum Exportieren gefunden")

# 2. Export der aggregierten Zeitmetriken (Wide-Format)
df_time_export = df[[
    'Participant_ID', 'Test', 'Attempt', 
    'accuracy_pct', 'correct_count', 'target_length',
    'avg_transition_time', 'median_transition_time', 
    'movement_variability_cv', 'avg_gap', 
    'avg_state_duration', 'overlap_count'
]].copy()

df_time_export.to_csv('aggregated_time_metrics.csv', index=False, encoding='utf-8-sig')
print(f"✅ Aggregierte Zeitmetriken gespeichert: 'aggregated_time_metrics.csv'")

# 3. Export der Zusammenfassungen
time_summary.to_csv('time_summary_by_test.csv', encoding='utf-8-sig')
participant_time_summary.to_csv('time_summary_by_participant.csv', encoding='utf-8-sig')
print(f"✅ Zusammenfassungen gespeichert: 'time_summary_by_test.csv' und 'time_summary_by_participant.csv'")

print("\n" + "="*100)
print("ZEITANALYSE ABGESCHLOSSEN")
print("="*100)
