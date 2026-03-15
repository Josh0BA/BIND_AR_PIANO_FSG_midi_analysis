# BIND_AR_PIANO_FSG_midi_analysis

Repository for MIDI-based piano motor-learning analysis (AR study context), including:

- state-transition timing analysis from MIDI sequences,
- finger dexterity sequence analysis,
- sleepiness scale (SSS) plotting and statistics.

## Overview

The project contains three analysis blocks:

1. `new_state_analysis/`
	- Detects predefined pitch-set states (`state0` to `state8`) from MIDI.
	- Computes transition metrics (for example `onset_to_onset`).
	- Exports transition-level CSV files and statistical summaries.
	- Creates learning-curve and pre/post plots.

2. `midi_finger_analysis/`
	- Reads finger test MIDI files.
	- Counts correct note sequence matches and keystrokes.
	- Exports participant-level dexterity table (`fingergeschicklichkeit.csv`).

3. `SleepynessScale/`
	- Loads SSS data from Excel or CSV.
	- Creates boxplots and descriptive statistics.
	- Runs repeated-measures statistics (ANOVA/Friedman + post-hoc).

## Repository Structure

BIND_AR_PIANO_FSG_midi_analysis/
|-- LICENSE
|-- README.md
|-- setup.py
|-- midi_finger_analysis/
|   |-- load_MIDI_finger.py
|   |-- statistical_analysis.py
|   `-- anova_fingerdex.py
|-- new_state_analysis/
|   |-- Main.py
|   |-- control.py
|   |-- Safe_check_order.py
|   |-- learning_curve_new.py
|   |-- pre_post_plot.py
|   |-- statistical_analysis_blocks.py
|   `-- statistical_analysis_pre_post.py
`-- SleepynessScale/
	 |-- Boxplots_SSS.py
	 `-- Statistical_analysis_SSS.py

## Requirements

## Python

- Recommended: Python 3.10+

## Core packages

- `pandas`
- `numpy`
- `pretty_midi`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scipy`

## Optional packages

- `openpyxl` (for reading `.xlsx` sleepiness files)
- `python-calamine` (fallback Excel engine used in SSS scripts)

## Setup (Windows PowerShell)

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy pretty_midi matplotlib seaborn statsmodels scipy openpyxl
```

`setup.py` is present, but direct `pip install ...` as above is the most reliable setup for the current scripts.

## Input Data Layout

### MIDI folder

Most MIDI scripts search for a folder named `Daten (MIDI)`.

Expected pattern:

```text
<repo-root>/
|-- Daten (MIDI)/
|   |-- <ParticipantID>/
|   |   |-- MIDI_<ParticipantID>_B1.mid
|   |   |-- MIDI_<ParticipantID>_B2.mid
|   |   |-- ...
|   |   |-- MIDI_<ParticipantID>_Pretest.mid
|   |   |-- MIDI_<ParticipantID>_Posttest.mid
|   |   `-- MIDI_<ParticipantID>_Fingertest1.mid
```

### Naming expectations

- State-transition pipeline expects file names like: `MIDI_<ParticipantID>_<Test>.mid`
- Finger pipeline processes only MIDI files whose name contains `finger`.

## Workflow A: State Transition Analysis

Run from repository root:

```powershell
python new_state_analysis/Main.py
python new_state_analysis/control.py
python new_state_analysis/Safe_check_order.py
python new_state_analysis/learning_curve_new.py
python new_state_analysis/pre_post_plot.py
python new_state_analysis/statistical_analysis_blocks.py
python new_state_analysis/statistical_analysis_pre_post.py
```

### Main outputs

`Main.py` writes (root):

- `sequence_length.csv`
- `state_sequence.csv`
- `last_found_idx.csv`
- `state_info_arrays.csv`

`control.py` writes (root):

- `sequence_analysis_results.csv`
- `transition_times_results.csv`

`Safe_check_order.py` writes (root):

- `safe_check_order.csv`

Plot scripts write into `Plots/` (created automatically):

- `learning_curve_all_h_s.png`
- `pre_post_boxplots.png`
- `pre_post_violin.png`

Statistical scripts write (root):

- Block-level:
  - `block_transition_time_wide.csv`
  - `block_transition_time_summary.csv`
  - `block_transition_time_log_wide.csv`
  - `block_transition_time_log_long.csv`
  - `block_transition_time_rm_anova_log_results.csv`
- Pre/Post-level:
  - `pre_post_transition_time_wide.csv`
  - `pre_post_transition_time_summary.csv`
  - `pre_post_transition_time_log_wide.csv`
  - `pre_post_transition_time_log_long.csv`
  - `pre_post_transition_time_rm_anova_log_results.csv`
  - `pre_post_transition_time_wilcoxon_results.csv`

## Workflow B: Finger Dexterity Analysis

Run from repository root:

```powershell
python midi_finger_analysis/load_MIDI_finger.py
python midi_finger_analysis/statistical_analysis.py
python midi_finger_analysis/anova_fingerdex.py
```

### Outputs

- `fingergeschicklichkeit.csv` (written next to `Daten (MIDI)` parent folder)
- additional console summaries and plots from analysis scripts

## Workflow C: Sleepiness Scale (SSS)

Place one of the following files in `SleepynessScale/`:

- `SleepinessScale_TN.xlsx` (preferred), or
- `SleepinessScale_TN.csv`

Run:

```powershell
python SleepynessScale/Boxplots_SSS.py
python SleepynessScale/Statistical_analysis_SSS.py
```

### Outputs

From `Boxplots_SSS.py`:

- interactive plots shown via matplotlib
- `SleepynessScale_deskriptive_statistik.csv`

From `Statistical_analysis_SSS.py`:

- `sleepiness_scale_descriptive_statistics.csv`
- `sleepiness_scale_inferential_results.csv`
- `sleepiness_scale_posthoc_results.csv`

## Notes and Troubleshooting

- If `transition_times_results.csv` is missing, run `new_state_analysis/Main.py` and `new_state_analysis/control.py` first.
- If Excel loading fails in SSS scripts, install `openpyxl`, or provide the CSV fallback file.
- Some scripts apply participant-specific manual corrections (for known special cases); check script contents before publication-grade analysis.
- Paths are mostly relative, so run commands from repository root for the most predictable output locations.

## License

See `LICENSE`.
