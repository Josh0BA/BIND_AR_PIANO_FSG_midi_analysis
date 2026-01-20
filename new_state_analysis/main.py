# Load Data -> MIDI -> Extract Pitches
import os
import re
from typing import Set
import pretty_midi
import pandas as pd
import numpy as np
from pyparsing import Dict

from midi_state_analysis.folder_utils import find_midi_data_folder

# ---------------------------------------------------------------------------
# 1. State-Definitionen (fixes Mapping deiner 9 Zustände)
# ---------------------------------------------------------------------------

STATE_DEFS: Dict[int, Set[int]] = {
    1: {65, 67, 72, 74, 77, 79},  # state1: F4, G4, C5, D5, F5, G5
    2: {60, 62, 64, 72, 77, 79},  # state2: C4, D4, E4, C5, F5, G5
    3: {60, 62, 64, 67, 72, 76},  # state3: C4, D4, E4, G4, C5, E5
    4: {60, 62, 64, 76, 77, 79},  # state4: C4, D4, E4, E5, F5, G5
    5: {62, 64, 65, 76, 77, 79},  # state5: D4, E4, F4, E5, F5, G5
    6: {64, 65, 72, 76, 77, 79},  # state6: E4, F4, C5, E5, F5, G5
    7: {60, 62, 72, 74, 76, 77},  # state7: C4, D4, C5, D5, E5, F5
    8: {64, 65, 72, 74, 77, 79},  # state8: E4, F4, C5, D5, F5, G5
    9: {62, 64, 65, 67, 72, 74},  # state9: D4, E4, F4, G4, C5, D5
}

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
        if filename.lower().endswith(('.mid', '.midi')) and 'finger' in filename.lower():
            file_path = os.path.join(dirpath, filename)

            try:
                # Extract participant ID and test/attempt from filename
                # New format: MIDI_{ParticipantID}_{TestAndAttempt}.mid
                # Example: MIDI_BE16MI_Fingertest1.mid (test name + attempt suffix)
                parts = filename.split('_')
                if len(parts) >= 3 and parts[0].upper() == 'MIDI':
                    participant_id = parts[1]
                    test_attempt = parts[2].split('.')[0]
                else:
                    # Fallback to legacy format: {ParticipantID}_{Appointment}_{Song}_{Attempt}.mid
                    participant_id = parts[0]
                    test_attempt = parts[2].split('.')[0] if len(parts) > 2 else 'Fingertest1'

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

                for i in range(0, len(played_notes) - window_size + 1, window_size):
                    window_notes = played_notes[i:i + window_size]
                    window_pitches = {note['pitch'] for note in window_notes}
                    
                    # Find matching state: check if all 6 expected state pitches are in the window
                    # (additional pitches are ignored)
                    best_match = None
                    
                    for state_num, state_pitches in STATE_DEFS.items():
                        # Subset matching: all state pitches must be in window pitches
                        if state_pitches.issubset(window_pitches):
                            best_match = state_num
                            break
                    
                    if best_match is not None:
                        detected_states.append({
                            'state': best_match,
                            'notes': window_notes,
                            'start_time': window_notes[0]['start'],
                            'end_time': window_notes[-1]['end'],
                            'window_index': i // window_size
                        })

# - When a state is recognized, save the state with associated pitches, start time, and end time
#
# Control -> Remove duplicate states, keeping only the last occurrence of consecutive identical states
# - Compare detected state sequence against expected sequence (from config)
# prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Test': test_name,
                    'Attempt': attempt,
                    'Keystrokes': len(played_notes),
                    # 'Correct_Sequences': correct_sequences,
                }

                data.append(info)

                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

# Analyze -> Compute transition times between states
