# Load Data -> MIDI -> Extract Pitches
import os
import re
from typing import Set, Dict, Tuple
import pretty_midi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_midi_data_folder(start_path: str = '.', target_folder: str = 'Daten (MIDI)'):
    """
    Sucht rekursiv nach dem MIDI-Datenordner und gibt dessen absoluten Pfad zurück.

    Args:
        start_path: Startpunkt der Suche
        target_folder: Name des gesuchten Ordners

    Returns:
        Absoluter Pfad zum gefundenen Ordner oder None
    """
    start_abs = os.path.abspath(start_path)

    # 1) Direkter Treffer im Startordner
    candidate = os.path.join(start_abs, target_folder)
    if os.path.isdir(candidate):
        return os.path.normpath(candidate)

    # 2) Rekursive Suche unterhalb des Startordners
    for dirpath, dirnames, _ in os.walk(start_abs):
        if target_folder in dirnames:
            return os.path.normpath(os.path.join(dirpath, target_folder))

    return None
    
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
                mismatch_positions = []
                found_state_ranges = []
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
                            
                            # Get indexes of these notes in played_notes
                            state_note_indices = [played_notes.index(n) for n in state_notes]
                            state_pitches = [n['pitch'] for n in state_notes]
                            
                            detected_states.append({
                                'state': target_state_id,
                                'notes': state_notes,
                                'start_time': state_notes[0]['start'],
                                'end_time': state_notes[-1]['end'],
                                'target_index': state_index, # Position in der Soll-Sequenz
                                'notes_skipped': i - current_note_idx,  # WICHTIG: Wie viele Noten waren "Müll" dazwischen?
                                'state_info': [target_state_id, state_pitches, state_note_indices]  # [state, pitches, indexes]
                            })
                            
                            # Setze den Zeiger hinter die letzte Note dieses erkannten States
                            # damit wir nicht dieselben Noten für den nächsten State nutzen
                            first_note_in_state = min(state_note_indices)
                            last_note_in_state = max(state_note_indices)
                            found_state_ranges.append(
                                f"State {target_state_id}: played_notes Index {first_note_in_state} to {last_note_in_state} (Position {state_index})"
                            )
                            current_note_idx = last_note_in_state + 1
                            found_state = True
                            break
                    
                        # else:
                        #     current_note_idx += 1
                        #     search_limit += 1

                    # if not found_state:
                    #     # Hier markieren wir explizit eine Lücke in der Datenstruktur
                    #     detected_states.append({
                    #         'state': None, 
                    #         'expected_state': target_state_id,
                    #         'target_index': state_index,
                    #         'error': 'MISSING'
                    #     })
                    #     # Wir setzen current_note_idx NICHT weiter, um dem nächsten State 
                    #     # die Chance zu geben, ab der gleichen Stelle gefunden zu werden.
                    #     mismatch_positions.append((target_state_id, state_index))

                if mismatch_positions:
                    last_found_pitch = None
                    last_found_idx = None
                    mismatch_output_lines = []
                    for state_entry in reversed(detected_states):
                        if state_entry.get('state') is not None and state_entry.get('notes'):
                            last_note = state_entry['notes'][-1]
                            last_found_pitch = last_note.get('pitch')
                            last_found_idx = played_notes.index(last_note)
                            break

                    for target_state_id, state_index in mismatch_positions:
                        if last_found_pitch is not None:
                            line = (
                                f"   ℹ️ State {target_state_id} an Position {state_index} nicht gefunden. "
                                f"Letzter Pitch: {last_found_pitch} (played_notes Index {last_found_idx})"
                            )
                        else:
                            line = (
                                f"   ℹ️ State {target_state_id} an Position {state_index} nicht gefunden. "
                                "Letzter Pitch: n/a"
                            )
                        print(line)
                        mismatch_output_lines.append(line)

                    if found_state_ranges:
                        header = "   🔎 Gefundene States (Index-Bereiche):"
                        # print(header)
                        mismatch_output_lines.append(header)
                        for entry in found_state_ranges:
                            line = f"   - {entry}"
                            # print(line)
                            mismatch_output_lines.append(line)

                    with open("state_mismatch_debug.txt", "a", encoding="utf-8") as debug_file:
                        debug_file.write(f"\n=== {participant_id} | {test_name} ===\n")
                        for line in mismatch_output_lines:
                            debug_file.write(line + "\n")

                # Calculate last found pitch and index for DataFrame
                last_found_pitch = None
                last_found_idx = None
                for state_entry in reversed(detected_states):
                    if state_entry.get('state') is not None and state_entry.get('notes'):
                        last_note = state_entry['notes'][-1]
                        last_found_pitch = last_note.get('pitch')
                        last_found_idx = played_notes.index(last_note)
                        break

                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Test': test_name, #example: Fingertest1, B2, Pretest
                    'detected_states': detected_states,
                    'last_found_pitch': last_found_pitch,
                    'last_found_idx': last_found_idx,
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
# Filter out Aufwärmen and Fingertest 1-4
df = df[~df['Test'].str.lower().str.contains('auf|finger', na=False)]
df['state_sequence'] = df['detected_states'].apply(lambda states: [d['state'] for d in states])
df['sequence_length'] = df['state_sequence'].apply(len)
df['state_info_arrays'] = df['detected_states'].apply(lambda states: [d.get('state_info', []) for d in states])
df[['Participant_ID', 'Test', 'sequence_length']].to_csv('sequence_length.csv', index=False)
df[['Participant_ID', 'Test', 'state_sequence']].to_csv('state_sequence.csv', index=False)
df[['Participant_ID', 'Test','last_found_pitch', 'last_found_idx']].to_csv('last_found_idx.csv', index=False)
df[['Participant_ID', 'Test', 'state_info_arrays']].to_csv('state_info_arrays.csv', index=False)
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
