from pathlib import Path

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

# Pfade relativ zum Projekt
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Daten (MIDI)"
OUTPUT_DIR = PROJECT_ROOT / "new_state_analysis" / "results"

def midi_to_note_name(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

def process_midi_to_csv(input_file: Path, output_dir: Path) -> None:
    """Liest eine MIDI-Datei und schreibt zwei CSVs ins Ausgabeverzeichnis."""
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
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    df_all.to_csv(output_dir / f"{stem}_alle_noten_chronologisch.csv", index=False)
    df_state_only.to_csv(output_dir / f"{stem}_nur_state_noten_chronologisch.csv", index=False)

    print(f"Verarbeitung abgeschlossen: {input_file.name}")
    print(f"Gesamtanzahl Noten: {len(df_all)}")
    print(f"Anzahl gefilterter State-Noten: {len(df_state_only)}")

if __name__ == "__main__":
    # Installation: pip install mido pandas
    midi_files = sorted(DATA_DIR.glob("*.midi")) + sorted(DATA_DIR.glob("*.mid"))
    if not midi_files:
        print(f"Keine MIDI-Dateien in {DATA_DIR} gefunden.")
    for midi_file in midi_files:
        process_midi_to_csv(midi_file, OUTPUT_DIR)
