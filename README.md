# BIND_AR_PIANO_FSG_midi_state_analysis
Analyse von MIDI-Dateien für das Motorik-lernen am Klavier mit AR. 

Anleitung:																													
-Installation, Lade das Projekt herunter oder klone das Repository.

-Daten vorbereiten
Erstelle einen Ordner mit dem Namen „Daten (MIDI)“. Lege darin Unterordner für jede Versuchsperson an, und speichere die jeweiligen MIDI-Dateien dort ab. Beispiel:
Daten (MIDI)/BE16MI/MIDI_BE16MI_B1.mid

-> Setup.py um benötigte bibliotheken zu installieren.

- New_state_analysis starten
Öffne ein Terminal und führe das Programm aus:
-> Main.py.
Das Programm sucht automatisch nach dem Ordner „Daten (MIDI)“.
Die Analyse erzeugt eine CSV-Datei.
Diese enthält pro Übergang: subject, block, state_from, state_to, onset-Zeiten, transition_time und die zugehörige Häufigkeit (h oder s).
- Um die Plots zu erstellen sind Lerning_curve_new.py und pre_post_plot.py auszuführen.
- für die statistische auswertung: statistical_analysis_blocks.py oder pre_post.py ausführen.

Funktionen:
- Erkennt 9 Zustände (state0–state8) anhand fixer 6er-Kombinationen von Pitches
- Berechnet Transitionen zwischen Zuständen mit Zeiten in Sekunden
- Ordnet Häufigkeiten (h/s) aus BLOCK_FREQ_BY_INDEX den Transitionen zu
- Automatische Pattern-Erkennung: Block_Training (B1-B8, 72 States) vs Test (Pre/Post, 54 States)
- Sucht automatisch nach "Daten (MIDI)" Ordner (aktuell/übergeordnet/rekursiv)
- Extrahiert Subject-ID aus Ordnername und Block aus Dateiname
- Schreibt alle Transitionen in CSV: subject, block, state_from/to, times, frequencies

Struktur: BIND_AR_PIANO_FSG_midi_state_analysis/                                                                        
BIND_AR_PIANO_FSG_midi_state_analysis/
├── midi_finger_analysis/
├── new_state_analysis/
├── README.md
├── LICENSE
└── setup.py                                         
