# cat_learn_automaticity

Longitudinal category-learning automaticity projects with
at-home training plus periodic lab sessions (behavior +
EEG).

This repository contains two closely-related study instances:

- **cat_learn_auto_pace_2025_s2** — Session 2 (Semester 2, MQ University; 2025)
- **cat_learn_auto_pace_2026_s1** — Session 1 (Semester 1, MQ University; 2026)

Both studies follow the same high-level structure:
- Participants complete repeated at-home training sessions (many days)
- Every ~5 days, participants come into the lab for a session measured with EEG (behavioral logs included here)
- Special at-home probe sessions include **dual-task** and **button-switch** manipulations (2025 s2 only)
- Special at-home probe sessions include **90 vs 180** category structure rotations as learning probes (2026 s1 only)

---

## Repository structure

- **cat_learn_auto_pace_2025_s2/**
  Complete session folder including code, data, DBM fits,
  and figures.

- **cat_learn_auto_pace_2026_s1/**
  Parallel session folder (similar codebase and intended
  analysis pipeline; data folders are currently
  placeholders).

Each session folder contains its own README with
session-specific notes.

---

## Analyses

Behavioral analysis is run from within a session folder, e.g.:

```bash
cd cat_learn_auto_pace_2025_s2/code
python inspect_results.py
```

Outputs (figures and cached model fits) are written to the
session’s `../figures/` and `../dbm_fits/` directories.

EEG analysis is currently minimal / in progress; the
existing `inspect_results_eeg.py` is not the primary
analysis entry point.
