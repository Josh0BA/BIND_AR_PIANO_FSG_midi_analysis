import mido
import csv
import os
from pathlib import Path

# 1. Definition deiner States (Notenkombinationen)
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

def export_midi_analysis(input_midi, output_csv):
    """
    Liest die MIDI-Datei aus und speichert erkannte States in eine CSV.
    """
    # MIDI laden
    mid = mido.MidiFile(input_midi)
    
    active_notes = set()
    results = []
    last_state = None
    total_time = 0

    for msg in mid:
        total_time += msg.time
        
        # Note On/Off Logik verarbeiten
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.add(msg.note)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            active_notes.discard(msg.note)

        # Prüfen, ob die aktuelle Kombination einem deiner States entspricht
        current_state = None
        for state_id, state_notes in STATE_DEFS.items():
            if active_notes == state_notes:
                current_state = state_id
                break
        
        # Speichern, wenn ein State erkannt wurde und es ein Wechsel ist
        if current_state is not None and current_state != last_state:
            results.append({
                'Zeit_Sekunden': round(total_time, 3),
                'State_ID': current_state,
                'MIDI_Pitches': sorted(list(active_notes))
            })
            last_state = current_state
        elif current_state is None:
            # Setzt den Status zurück, wenn keine gültige Kombi mehr klingt
            last_state = None

    # CSV Export
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Zeit_Sekunden', 'State_ID', 'MIDI_Pitches'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Erfolg! {len(results)} States wurden in '{output_csv}' gespeichert.")
    else:
        print("Keine passenden States in der MIDI-Datei gefunden.")

if __name__ == "__main__":
    # Hinweis: Du musst mido installieren: pip install mido
    
    # Pfad zum MIDI-Ordner
    midi_folder = Path(__file__).parent.parent / "Daten (MIDI)"
    output_folder = Path(__file__).parent / "results"
    
    # Erstelle Ausgabeordner falls nicht vorhanden
    output_folder.mkdir(exist_ok=True)
    
    # Verarbeite alle MIDI-Dateien im Ordner
    midi_files = list(midi_folder.glob("*.midi")) + list(midi_folder.glob("*.mid"))
    
    if not midi_files:
        print(f"Keine MIDI-Dateien in '{midi_folder}' gefunden.")
    else:
        print(f"{len(midi_files)} MIDI-Dateien gefunden. Starte Analyse...\n")
        
        for midi_file in midi_files:
            output_csv = output_folder / f"{midi_file.stem}_analysis.csv"
            print(f"Verarbeite: {midi_file.name}")
            export_midi_analysis(str(midi_file), str(output_csv))
            print()
