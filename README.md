# SMS-Based AI Teaching Assistant for Upper Primary Pupils

Course project for 04-652 AI System Design.

## 1. Project

Many upper-primary pupils in Rwanda have no smartphone or mobile data, so app-based revision tools do not reach them
(SDG 4, Quality Education). The SMS tutor works on any basic phone. It sends a Science and Technology question taken
from the REB Pupil's Book, receives the pupil's text reply, and decides whether the reply is correct.

The hard part is the decision. Exact string matching rejects correct answers that are reworded, misspelt, or typed in
SMS shorthand. This project replaces it with a language model plus a classifier that compares the reply with the official
answer. The repo holds the data work for that system: the questions, the extraction script, the preprocessing strategy
and pipeline, and the model design.

**Task type:** binary classification. **Target Y:** `is_correct` (1 = the reply is correct for the item, 0 = it is not).

## 2. Dataset and features

**Source.** Rwanda Basic Education Board (REB) Pupil's Books and Teacher's Guides, 2026-2027 editions
(https://elearning.reb.rw): P4 to P6 Science and Elementary Technology, plus a P4 English guide. Questions are extracted
with regex rules in `src/extract_items.py`, with no manual retyping. Answer keys come from the Teacher's Guides where
supplied. No source contains pupil replies.

**Two tables.**

| Table | Rows (N) | Columns (P) | State |
|---|---|---|---|
| Raw item table, `data/raw/items.csv` | 965 (P4 387, P5 149, P6 429) | 28 | built; 678 rows have an official answer |
| Modelling table, one row per (item, candidate reply) | about 2,152 (538 items x 4 replies) | roughly 25 to 30 after encoding | planned |

The 538 items are the Science rows that have a question text and an answer, need no picture or computer, and have an
answer of at most 160 characters (it must fit in one SMS reply): P4 201, P6 337. Each item gets four candidate replies
written by a language model under a fixed specification: one correct reply, written as a pupil would send it, and three
wrong ones (a typical misconception, a partly right reply, an off-topic reply). That gives 25% positive, an imbalance
ratio of 3:1. A separate real-life set, labelled by people, is kept for the final test.

**Features of the raw item table** (full definitions in `data/raw/README.md`):

| Group | Columns | Type | Use in the model |
|---|---|---|---|
| Identity | `item_id` | identifier | grouping key only |
| Where the item sits | `subject`, `class_level`, `section`, `unit_no`, `unit_title`, `q_no` | nominal; `class_level` is ordinal | `subject`, `section` one-hot; `class_level` ordinal (4 < 5 < 6); the other three not used |
| Item format | `item_type` (MCQ, TF, FILL, MATCH, OPEN), `n_options`, `n_blanks`, `n_subitems` | nominal; discrete counts | one-hot; counts scaled |
| Item length | `n_words_stem`, `sms_chars`, `sms_segments` | discrete counts, right-skewed | winsorized, `log1p`, standardised, after the collinearity screen |
| SMS suitability | `needs_visual`, `needs_computer`, `uses_unicode`, `sensitive_topic` | binary | the first two filter rows; `uses_unicode` is a feature; `sensitive_topic` is constant in this extract |
| Text | `stem`, `options`, `word_bank`, `option_labels`, `sms_text` | text / JSON lists | used to build features |
| Official answer | `answer_key`, `answer_text` | text | source of the label and of the similarity features, not a feature itself |
| Extraction metadata | `answer_key_source`, `stem_source`, `review_flag` | nominal | excluded, because they would leak whether an answer exists |

**Features of the modelling table (planned).** The item features above, plus features that compare the reply with the
official answer: embedding cosine similarity, token overlap, edit-distance ratio, number match, length ratio and the
reply word count. The reply-to-answer features are expected to carry most of the signal, because every item has the same
1 correct to 3 wrong mix, so item-level columns cannot predict correctness on their own.

**Preprocessing.** One scikit-learn `Pipeline` / `ColumnTransformer` (`src/pipeline_builder.py`, custom steps in
`src/transformers.py`), fitted on the training split only. Missing answers and question texts are never imputed: those
rows are kept in the raw table and left out of training. Outliers are capped, not deleted (IQR fence for length features,
99th percentile for zero-heavy counts). Correlated length features are dropped with a pairwise rule (|r| >= 0.85) and a
VIF > 10 rule. The reasoning behind each choice, with the numbers, is in
`phase1_strategy/Phase1_Strategy_Report.pdf`.

## 3. Model

**Status.** Designed in Phase 1, to be trained in Phase 2. This repo does not report any trained-model results yet.

**Input and output.** For one (item, reply) pair the model reads the preprocessed item features and the reply-to-answer
features, and returns the probability that the reply is correct. The tutor accepts the reply when the probability passes
a tuned threshold, and otherwise asks the pupil to try again.

**Models compared.**

| Model | Role |
|---|---|
| Exact / fuzzy string match | baseline: what the project is meant to beat |
| Logistic regression, `class_weight="balanced"` | simple, interpretable baseline on the engineered features |
| Gradient boosting | non-linear model; can use the item features as modifiers of the similarity needed |

The sentence-embedding model that produces the cosine similarity is pretrained and used frozen, so it is never fitted on
this data.

**Evaluation.**
- **Split:** 70 / 15 / 15 by item (all four replies of one item stay together), with 5-fold `StratifiedGroupKFold` inside
  the training part for tuning. Items with the same question text are merged into one group.
- **Metrics:** precision, recall, F1 and PR-AUC for the correct class, not accuracy. Accepting a wrong reply is worse
  than asking a pupil to retry, so the decision threshold is tuned on the validation part to favour precision.
- **Real-life test:** replies from real people, labelled by two raters, used once at the end and never for tuning.
  This checks that the model has not just learned the style of the language-model replies.

**Known risk.** Language-model mistakes may not look like real pupil mistakes. The real-life set is the control for this.

## Status

| Item | State |
|---|---|
| Item extraction (P4, P5, P6 Science; P4 English guide) | done, 965 rows; known fixes listed in the Phase 1 report, section 5 |
| Phase 1 strategy report | done: `phase1_strategy/Phase1_Strategy_Report.pdf` |
| P5 answer keys | blocked: P5 Teacher's Guide not yet supplied |
| Reply generation (`data/raw/replies.csv`) | not started |
| Phase 2 pipeline, 8 figures, final report | not started |

## Quick start

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/extract_items.py
python phase1_strategy/phase1_audit.py     # reprints every number in the Phase 1 report
```

Run it from the repo root. It reads every `.pdf` and `.txt` file in `data/raw/source_books/` and writes
`data/raw/items.csv` and `data/raw/extraction_report.csv`. Check the report after each run: a book with 0 items or an
odd split of item types needs a rule fix in the extractor.

To use other folders: `python src/extract_items.py <book_folder> <output.csv>`.

PDFs are read with poppler's `pdftotext` if it is installed, otherwise with `pypdf`. `.txt` exports always work.
Optional: `python -m pytest` runs a smoke test of the extractor.

## Folder layout

```
data/
  raw/
    source_books/        REB Pupil's Books (.pdf git-ignored, .txt tracked)
    items.csv            one row per end-of-unit question (generated)
    extraction_report.csv  one row per book (generated)
    README.md            how extraction works and every column
  processed/             pipeline outputs (Phase 2)
src/
  extract_items.py       book files -> items.csv
  transformers.py        custom scikit-learn transformers (Phase 2)
  pipeline_builder.py    preprocessing pipeline (Phase 2)
phase1_strategy/         Phase 1 report and the script that reproduces its numbers (phase1_audit.py)
phase2_implementation/   notebook and final report
figures/                 figures for the reports
tests/                   smoke test for the extractor
PROJECT_LOG.md           decisions and changes, newest first
```

## Known limits

- Only end-of-unit assessments are read, not in-lesson activities.
- The P4 Science Pupil's Book parse returned 8 of 12 units, and `unit_title` is unreliable (see the Phase 1 report, section 5).
- Matching questions (2 so far) stay as one row and are not split into parts.
- `src/transformers.py` and `src/pipeline_builder.py` are Phase 2 stubs, with column lists to be fixed by the Phase 1 plan.
