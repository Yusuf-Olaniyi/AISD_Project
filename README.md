# SMS AI Tutor — Adaptive Scheduler: Data Preprocessing

**Course:** 04-652 Artificial Intelligence System Design — Assignment 2 (Data Preprocessing)

## Problem Statement

Predict whether a pupil will answer a given Science & Elementary Technology (SET) revision
item correctly on their next attempt, using that pupil's response history (prior accuracy,
time since last attempt, attempt number, sub-topic). This prediction drives the **adaptive
mastery scheduler** in the SMS AI Tutor system design: the scheduler uses it to prioritize
which item to resurface next over SMS, personalizing revision order for Upper Primary pupils
in Rwanda who access the tutor over basic feature phones (see the project's system design
proposal, Section III — AI System Design and Task Formulation).

## Dataset Overview

- **Source:** Synthetic simulation (`data/raw/pupil_response_log.csv`), generated to carry
  realistic structure — genuine MCAR/MAR/MNAR missingness mechanisms, skewed latency
  distributions, and outliers — in the absence of real pilot data. No live pilot has been
  run yet (see the design proposal's Limitations section); this dataset stands in for what
  the scheduler's real training log will look like once deployed.
- **Sample size (N):** 4,947 response-event rows, spanning 220 simulated pupils, 88 items,
  and 11 REB SET sub-topics.
- **Feature count (P):** 9 raw columns (3 identifiers, 5 predictive features, 1 target).
- **Target variable (Y):** `is_correct` — binary (1 = correct, 0 = incorrect or SMS timeout).

## Task Type

**Binary Classification.**

## Repository Structure

See `phase1_strategy/Phase1_Strategy_Report.pdf` for the full preprocessing audit and
strategy (missingness mechanism diagnosis, outlier rules, encoding/scaling plan, and
leakage-prevention architecture). Phase 2 (pipeline implementation + diagnostic figures)
is tracked under `phase2_implementation/`.

```
.
├── data/
│   ├── raw/                 # pupil_response_log.csv (synthetic, see Phase 1 report)
│   └── processed/           # populated in Phase 2
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   ├── notebooks/           # pipeline_execution.ipynb (Phase 2)
│   └── Phase2_Final_Report.pdf   (Phase 2)
├── src/
│   ├── transformers.py      # custom sklearn transformers (stubs; implemented in Phase 2)
│   └── pipeline_builder.py  # ColumnTransformer / Pipeline definitions (stubs)
└── figures/                 # 8 diagnostic figures (Phase 2 deliverable)
```

## Environment

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

## Reproduction

1. `data/raw/pupil_response_log.csv` is the raw, untouched dataset.
2. Phase 1 strategy (what will be done and why) is documented in
   `phase1_strategy/Phase1_Strategy_Report.pdf`.
3. Phase 2 (due Oct 18, 2026) will implement that strategy in
   `phase2_implementation/notebooks/pipeline_execution.ipynb` using the modular
   transformers in `src/`, and report any strategy-vs-reality deviations in
   `phase2_implementation/Phase2_Final_Report.pdf`.
