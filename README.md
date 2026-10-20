# SMS AI Tutor

Course project for 04-652 AI System Design.

An SMS-based tutor gives upper-primary learners in Rwanda science and technology revision on basic feature phones
(SDG 4, Quality Education). It sends a question from the REB Pupil's Book, reads the pupil's reply, and checks whether
the reply is correct. This repo holds the data work for that system: the real questions, the extraction script,
the Phase 1 strategy and, later, the Phase 2 preprocessing pipeline.

## Status

| Item | State |
|---|---|
| Item extraction from P4-P6 SET | done |
| Answer keys / "is the student correct" label | to decide (see `PROJECT_LOG.md`) |
| Phase 1 strategy report | done|
| Phase 2 pipeline, 8 figures, final report | not started |

## Quick start

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/extract_items.py
```

Run it from the repo root. It reads every `.pdf` and `.txt` file in `data/raw/raw_pdfs/` and writes
`data/raw/items.csv` and `data/raw/extraction_report.csv`. Check the report after each run: a book with 0 items or an
odd split of item types needs a rule fix in the extractor.

To use other folders: `python src/extract_items.py <book_folder> <output.csv>`.

PDFs are read with poppler's `pdftotext` if it is installed; otherwise with `pypdf`. `.txt` exports always work.
Optional: `python -m pytest` runs a smoke test of the extractor.

## Folder layout

```
data/
  raw/
    raw_pdfs/        REB Pupil's Books (.pdf git-ignored, .txt tracked)
    items.csv            one row per end-of-unit question (generated)
    extraction_report.csv  one row per book (generated)
    README.md            how extraction works and every column
  processed/             pipeline outputs (Phase 2)
src/
  extract_items.py       book files -> items.csv
  transformers.py        custom scikit-learn transformers (Phase 2)
  pipeline_builder.py    preprocessing pipeline (Phase 2)
phase1_strategy/         Phase 1 report
phase2_implementation/   notebook and final report
figures/                 figures for the reports
tests/                   smoke test for the extractor
PROJECT_LOG.md           decisions and changes, newest first
```

## Data

Source: Rwanda Basic Education Board (REB), Science and Elementary Technology Pupil's Book, Primary 6,
end-of-unit assessments. Items are extracted with regex rules, with no manual retyping and no language model.
The book contains no answer keys and no student responses, so `answer_key` is empty for now.
Column definitions are in `data/raw/README.md`.

## Known limits

- Only end-of-unit assessments are read, not in-lesson activities.
- Matching questions (2 so far) stay as one row and are not split into parts.
- `src/transformers.py` and `src/pipeline_builder.py` are Phase 2 stubs, with column lists to be fixed by the Phase 1 plan.
