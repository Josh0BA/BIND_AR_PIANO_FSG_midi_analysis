import mido
import pandas as pd

# 1. Deine State-Definitionen (Pitches)
STATE_DEFS = {
    0: {65, 67, 72, 74, 77, 79},
    1: {60, 62, 64, 72, 77, 79},
    2: {60, 62, 64, 67, 72, 76},
    3: {60, 62, 64, 76, 77, 79},
    4: {62, 64, 65, 76, 77, 79},
    5: {64, 65, 72, 76, 77, 79},
    6: {60, 62, 72, 74, 76, 77},
    7: {64, 65, 72, 74, 77, 79},
    8: {62, 64, 65, 67, 72, 74}
}

# Erzeuge ein Set aller erlaubten Pitches aus den States
ALLOWED_PITCHES = set().union(*STATE_DEFS.values())

def midi_to_note_name(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

def process_midi_to_csv(input_file):
    mid = mido.MidiFile(input_file)
    
    all_notes = []
    abs_tick = 0

    # Iteriere durch alle Nachrichten in der Datei
    for msg in mid:
        # Ticks aufsummieren (mido.MidiFile liefert Delta-Ticks)
        abs_tick += msg.time
        
        # Wir interessieren uns nur für Note_On Ereignisse (Anschläge)
        if msg.type == 'note_on' and msg.velocity > 0:
            note_data = {
                "Tick": int(abs_tick),
                "MIDI_Pitch": msg.note,
                "Note_Name": midi_to_note_name(msg.note)
            }
            all_notes.append(note_data)

    # Erstelle DataFrames
    df_all = pd.DataFrame(all_notes)
    
    # Filter-Logik: Nur Noten, die in einem der States vorkommen
    df_state_only = df_all[df_all['MIDI_Pitch'].isin(ALLOWED_PITCHES)].copy()

    # Export
    df_all.to_csv('alle_noten_chronologisch.csv', index=False)
    df_state_only.to_csv('nur_state_noten_chronologisch.csv', index=False)
    
    print(f"Verarbeitung abgeschlossen.")
    print(f"Gesamtanzahl Noten: {len(df_all)}")
    print(f"Anzahl gefilterter State-Noten: {len(df_state_only)}")

if __name__ == "__main__":
    # Installation: pip install mido pandas
    process_midi_to_csv('MIDI_BE13RE_B1.mid')
