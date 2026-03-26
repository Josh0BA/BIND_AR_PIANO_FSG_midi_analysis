import os
import re
import pretty_midi
import pandas as pd
import numpy as np

from midi_state_analysis.folder_utils import find_midi_data_folder

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

                # Split test name and attempt (attempt expected as trailing digits)
                match = re.match(r"([A-Za-z]+)(\d+)", test_attempt)
                if match:
                    test_name = match.group(1)
                    attempt = match.group(2)
                else:
                    test_name = test_attempt
                    attempt = '1'

                # Extract note keystrokes

                # Load MIDI file
                midi = pretty_midi.PrettyMIDI(file_path)

                # Remove notes that start after 30 seconds
                for instrument in midi.instruments:
                    instrument.notes = [note for note in instrument.notes if note.start <= 30.0]

                # Your expected pattern (adjust to match what they were supposed to play)
                expected_sequence = ['F4', 'C4', 'E4', 'D4', 'F4']
                sequence_len = len(expected_sequence)

                # Extract played notes (non-drum, sorted by start time)
                played_notes = []
                for instrument in midi.instruments:
                    if not instrument.is_drum:
                        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
                        for note in sorted_notes:
                            name = pretty_midi.note_number_to_name(note.pitch)
                            played_notes.append(name)

                        #to get sequence of a specific participant for example to fix if they played the wrong sequence
                        # if (participant_id == "JE13CL"):
                        #     print(f"Played notes for {participant_id}: {played_notes}")
                        # if (participant_id == "LN15CH"):
                        #     print(f"Played notes for {participant_id}: {played_notes}")
                        # if (participant_id == "LO07EV"):
                        #     print(f"Played notes for {participant_id}: {played_notes}")

                # Count correct sequences using a sliding window
                correct_sequences = 0
                for i in range(len(played_notes) - sequence_len + 1):
                    if played_notes[i:i+sequence_len] == expected_sequence:
                        correct_sequences += 1


                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Test': test_name,
                    'Attempt': attempt,
                    'Keystrokes': len(played_notes),
                    'Correct_Sequences': correct_sequences,
                }

                data.append(info)

                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")


df_finger = pd.DataFrame(data)

pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # No limit on display width
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents

# Create label column for wide format - map to Fingertest1-4
label_mapping = {
    'Finger_1-1': 'Fingertest1', 
    'Finger_1-2': 'Fingertest2', 
    'Finger_5-1': 'Fingertest3', 
    'Finger_5-2': 'Fingertest4' 
}
df_finger['label'] = df_finger['Test'] + '_' + df_finger['Attempt'].astype(str)

# Pivot to get one row per participant
df_correct = df_finger.pivot(index='Participant_ID', columns='label', values='Correct_Sequences')
df_keys = df_finger.pivot(index='Participant_ID', columns='label', values='Keystrokes')
df_ratio = df_correct * 5 / df_keys

# rename columns for clarity
df_correct.columns = [f"{col}_correct" for col in df_correct.columns]
df_keys.columns = [f"{col}_keys" for col in df_keys.columns]
df_ratio.columns = [f"{col}_ratio" for col in df_ratio.columns]


#combine the dataframes
df_combined = pd.concat([df_correct, df_keys, df_ratio], axis=1)
df_combined.reset_index(inplace=True)

# Add Timepoint column for Pretest vs Posttest grouping
# Pretest: Fingertest1, Fingertest2 
# Posttest: Fingertest3, Fingertest4 
for col in df_combined.columns: 
    if 'Fingertest1' in col or 'Fingertest2' in col: 
        timepoint_col = col.replace('_correct', '_timepoint').replace('_keys', '_timepoint').replace('_ratio', '_timepoint') 
        if timepoint_col not in df_combined.columns: 
            df_combined[timepoint_col] = 'Pretest' 
    elif 'Fingertest3' in col or 'Fingertest4' in col: 
        timepoint_col = col.replace('_correct', '_timepoint').replace('_keys', '_timepoint').replace('_ratio', '_timepoint') 
        if timepoint_col not in df_combined.columns: 
            df_combined[timepoint_col] = 'Posttest' 

# Save CSV in the same directory as the MIDI data folder
output_path = os.path.join(os.path.dirname(root_folder), "fingergeschicklichkeit.csv")
df_combined.to_csv(output_path, index=False)

print(f"✓ Analyse abgeschlossen: {output_path}")
print(df_combined)