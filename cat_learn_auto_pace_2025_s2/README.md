# cat_learn_auto_pace_2025_s2

2025 Session 2 (Semester 2, MQ University): longitudinal
at-home category learning with periodic lab visits (behavior
+ EEG).

## Directory structure

- **code/**
  - `inspect_results.py` — main behavioral analysis script (figures + DBM fits)
  - `util_func_dbm.py` — decision-bound model likelihoods and DBM fitting routine
  - `run_exp.py` — experiment runtime code
  - `inspect_results_eeg.py` — EEG-related analysis (currently limited / in progress)

- **data/**
  At-home training logs, organized by participant folder:
  `data/subj_<hash>/sub_<ID>_day_<NN>_data.csv`

  Includes participant-specific irregularities (extra days, mislabels) that are handled in `inspect_results.py`.

- **data_lab_behave/**
  Lab behavioral logs as flat CSV files:
  `data_lab_behave/sub_<ID>_day_<NNN>_data.csv`

- **data_lab_eeg/**
  Raw EEG files (e.g., `.bdf`) for lab sessions.

- **dbm_fits/**
  Cached DBM fitting results:
  - `dbm_results.csv` (generated automatically if missing)

- **figures/**
  Output figures created by `inspect_results.py`

---

## How to run

From `code/`:

```bash
python inspect_results.py
```

This will:

1. Load and clean at-home + lab behavioral data (including several hard-coded subject/day fixes)

2. Apply inclusion criteria:

   * keep only subjects with data in all session types (training, dual-task, button-switch, lab)
   * Stroop (ns_*) accuracy >= 0.80 where available

3. Fit decision-bound models (DBM) per subject × day if `../dbm_fits/dbm_results.csv` does not exist

4. Generate figures into `../figures/`

---

## Session types & day coding (as used in analysis)

The analysis script organizes data into four session types:

* **Training at home**
  Days excluding [22, 23, 24] (after subject-wise day re-indexing)

* **Dual-Task at home**
  Day 22

* **Button-Switch at home**
  Days 23 and 24

* **Training in the Lab**
  Lab sessions are mapped onto the at-home day axis for plotting:
  `{1: 0.5, 2: 4.5, 3: 8.5, 4: 12.5, 5: 21}`

Day values are then re-ranked within-subject for a common plotting index.

---

## Decision-bound modeling (DBM)

DBMs are fit per **subject × day** with `block_size = 25` trials.

Models:

* Unidimensional bound on x (two side variants)
* Unidimensional bound on y (two side variants)
* General linear classifier (GLC; two side variants)

Model selection:

* Best model = minimum BIC per subject × day
* Best model class collapsed to:

  * **procedural** = GLC
  * **rule-based** = unidimensional

Results are cached in `../dbm_fits/dbm_results.csv` to avoid re-fitting on subsequent runs.

---

## Reproducibility notes

* The at-home loader includes a hard-coded “day exclusion
  list” and a set of subject-specific fixes for mislabeled
  days/subjects. See inline comments in
  `code/inspect_results.py`.

## Data files: trial-level CSV structure

At-home training (and related at-home probe sessions) are stored as trial-level CSV files.
Each row corresponds to **one trial**.

Example filename:
- `sub_017_day_01_data.csv`

### Columns

| Column   | Type | Description |
|---------|------|-------------|
| `subject` | int | Participant ID (numeric). |
| `day`     | int | Training day index within participant (starts at 1 for at-home training; special probe days may be coded as 22–24 depending on the session). |
| `trial`   | int | Trial index within day (0-based in the raw files). |
| `cat`     | str | Ground-truth category label for the stimulus (e.g., `"A"` or `"B"`). |
| `x`       | float | Stimulus value on dimension X in the **original stimulus space** (typically 0–100). |
| `y`       | float | Stimulus value on dimension Y in the **original stimulus space** (typically 0–100). |
| `xt`      | float | Transformed/scaled version of `x` used for modeling/decision-bound fits (task-specific transform). |
| `yt`      | float | Transformed/scaled version of `y` used for modeling/decision-bound fits (task-specific transform). |
| `resp`    | str | Participant response label (e.g., `"A"` or `"B"`). |
| `rt`      | float | Reaction time in milliseconds (ms). |
| `fb`      | str | Feedback text shown to the participant (e.g., `"Correct"` / `"Incorrect"`). |

### Notes / conventions

- **Accuracy** is typically computed as `acc = (cat == resp)` (1 = correct, 0 = incorrect).
- Trials may be aggregated into blocks for analysis (e.g., blocks of 25 trials) and averaged by subject × day.
- Some sessions include additional columns (e.g., Stroop/dual-task variables such as `ns_*`) on specific probe days; these columns will be absent (`NaN` after merge) for standard training trials.

