# UFC Predictions

`ufcpred` is a leakage-resistant UFC fight winner probability project. It ingests the
[UFC-DataLab](https://github.com/komaksym/UFC-DataLab) historical dataset, normalises it
into SQLite, calculates pre-fight Elo ratings and point-in-time career statistics, and
trains chronological logistic-regression and LightGBM models.

The authoritative design and phase gates are in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). This README is the operational guide
for running and maintaining the local project.

## Current status

| Phase | Status | Output |
|---|---|---|
| 0 — project and database schema | Complete | `data/ufc.db` core tables |
| 1 — UFC-DataLab ingestion | Complete | raw fight and fighter dataframes |
| 2 — SQLite normalisation | Complete | events, fighters, fights, fight stats |
| 3 — Elo ratings | Complete | `fight_elo` table |
| 4 — point-in-time features | Complete | `features` table |
| 5 — model training | Complete | `models/model.pkl`, reports |
| 6 — named-matchup CLI | Complete | `ufcpred.predict` command |
| 7 — automated incremental updates | Not implemented yet | planned update command |

The current validation-selected model is LightGBM. Its test log loss is `0.6637` and
test accuracy is `60.54%`. Logistic regression scored `0.6535` and `61.40%` on the same
untouched test period, but the test set was not used to change the selected model.

## Data preservation

The working dataset, vendored source, trained models, and reports are intentionally
gitignored. A Git clone alone is **not a backup** of these files:

- `data/`
- `vendor/`
- `models/`
- `reports/`
- `.venv/`

Do not delete the database, source CSVs, models, reports, or provenance metadata during
routine maintenance. Before refreshing data or retraining, make timestamped copies as
described in [Dataset maintenance](#dataset-maintenance). The training command preserves
existing model and report artifacts with timestamped names before writing new versions.

## Requirements

- Python 3.11 or newer; the current environment uses Python 3.12.
- Git.
- PowerShell for the commands below.
- Enough local disk space for the virtual environment, vendored dataset, SQLite database,
  and model artifacts.

All Python dependencies must stay inside the repository-local `.venv`. Do not install
project packages globally.

## First-time setup

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Clone the upstream dataset locally:

```powershell
git clone --depth 1 https://github.com/komaksym/UFC-DataLab.git vendor/UFC-DataLab
git -C vendor/UFC-DataLab rev-parse HEAD
```

The dataset used for the current results came from upstream commit:

```text
3268146c05211de9deab8b9b4c0bb4a954815f0b
```

Record the actual checked-out hash in `data/ingest_meta.json`. Dataset revisions can
change results, so preserve that file alongside database backups.

## Build the complete current pipeline

Run the phases in order. Do not train directly from UFC-DataLab's processed CSVs.

### 1. Initialise the database

```powershell
.\.venv\Scripts\python.exe -c "from ufcpred.db import init_db; init_db()"
```

### 2. Check the raw data

```powershell
.\.venv\Scripts\python.exe -c "from ufcpred.ingest import load_raw; fights, fighters = load_raw(); print('fights:', len(fights), 'fighters:', len(fighters), 'latest:', fights['event_date'].iloc[0])"
```

Approximately 8,700 fights and at least 3,700 fighter-detail rows are expected. The
latest event should be recent relative to the upstream checkout.

### 3. Normalise the source into SQLite

```powershell
.\.venv\Scripts\python.exe -m ufcpred.parse
```

This populates `events`, `fighters`, `fights`, and `fight_stats` in `data/ufc.db`.
Warnings and known name-key collisions are written to `data/ingest_warnings.log`.

### 4. Calculate chronological Elo ratings

```powershell
.\.venv\Scripts\python.exe -m ufcpred.ratings
```

This rebuilds the `fight_elo` table and prints the current top 15 ratings. Stored values
are pre-fight ratings; the current fight result is never included in its own Elo input.

### 5. Build leakage-safe features

```powershell
.\.venv\Scripts\python.exe -m ufcpred.features
```

This materialises one `features` row per fight with a known winner. Career aggregates use
only fights on strictly earlier dates. A fixed-seed fighter-order swap prevents the model
from learning the source dataset's red-corner position.

Expected current checks:

- 8,581 feature rows.
- Label mean close to 0.50.
- No feature more than 60% NULL.

### 6. Train and evaluate the models

```powershell
.\.venv\Scripts\python.exe -m ufcpred.train
```

The split is chronological:

- training: before 2023-01-01;
- validation: calendar year 2023;
- test: 2024-01-01 onward.

The command trains coin-flip and Elo baselines, logistic regression, and LightGBM. It
uses the validation partition for early stopping, model selection, and any calibration
decision. The test partition is reserved for final reporting.

Generated files:

- `models/model.pkl` — persisted model, feature order, training medians, and metadata;
- `reports/metrics.md` — validation and test metrics;
- `reports/calibration.png` — ten-bin test calibration plot;
- `reports/feature_importance.md` — LightGBM gain importance.

Inspect the current result:

```powershell
Get-Content reports\metrics.md
Get-Content reports\feature_importance.md
Invoke-Item reports\calibration.png
```

Inspect the model artifact without making a prediction:

```powershell
.\.venv\Scripts\python.exe -c "import joblib; artifact = joblib.load('models/model.pkl'); print(artifact['model_type']); print(artifact['trained_at_utc']); print(artifact['feature_columns'])"
```

## Predicting a matchup

Supply two case-insensitive exact fighter names and an optional matchup date:

```powershell
.\.venv\Scripts\python.exe -m ufcpred.predict "Fighter One" "Fighter Two" --date 2026-08-01
```

For example:

```powershell
.\.venv\Scripts\python.exe -m ufcpred.predict "Jon Jones" "CM Punk" --date 2026-08-01
```

The command resolves both names, reconstructs career and Elo state using fights strictly
before the requested date, and prints complementary win probabilities. It evaluates both
fighter orientations and averages them so reversing the two command-line names cannot
change the underlying matchup probabilities. A warning is printed if either fighter has
fewer than three prior UFC fights.

Omit `--date` to use today's date. Use `--model <path>` to inspect a preserved model
artifact instead of `models/model.pkl`.

## Running tests

Run only the project's tests because `vendor/UFC-DataLab` contains its own unrelated test
suite and is inside the working tree:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The current expected result is 30 passing tests. Bare `pytest` currently descends into
the vendored repository and can fail during collection; repository-level pytest exclusion
will be added during final wrap-up.

## Dataset maintenance

Phase 7 will automate safe incremental refreshes. Until then, refresh the dataset
manually and deliberately.

### 1. Preserve the current state

Create timestamped copies before touching the upstream checkout or database:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force data\archive | Out-Null
Copy-Item data\ufc.db "data\archive\ufc-$stamp.db"
Copy-Item data\ingest_meta.json "data\archive\ingest-meta-$stamp.json"
git -C vendor/UFC-DataLab rev-parse HEAD | Out-File "data\archive\vendor-commit-$stamp.txt"
```

These commands copy data; they do not remove the current files.

### 2. Inspect upstream changes before applying them

```powershell
git -C vendor/UFC-DataLab fetch origin
git -C vendor/UFC-DataLab log --oneline HEAD..origin/main
git -C vendor/UFC-DataLab diff --stat HEAD origin/main
```

Review the diff, especially deleted or renamed CSVs and scraper changes. Only after that
review, fast-forward the local vendored checkout:

```powershell
git -C vendor/UFC-DataLab merge --ff-only origin/main
git -C vendor/UFC-DataLab rev-parse HEAD
```

Update `data/ingest_meta.json` with the new hash and refresh date.

### 3. Rebuild downstream state

After an upstream change, run:

```powershell
.\.venv\Scripts\python.exe -m ufcpred.parse
.\.venv\Scripts\python.exe -m ufcpred.ratings
.\.venv\Scripts\python.exe -m ufcpred.features
.\.venv\Scripts\python.exe -m ufcpred.train
.\.venv\Scripts\python.exe -m pytest tests -q
```

Then compare row counts and `reports/metrics.md` with the preserved version. Investigate
unexpected count decreases, old maximum event dates, large metric changes, or new warning
categories before treating the refresh as accepted.

## Data leakage rules

These rules are mandatory for every new feature or model:

1. A feature for a fight may use only bouts on dates strictly earlier than that fight.
2. Fit imputation, scaling, calibration, and model-selection decisions without test data.
3. Never model from `stats_processed.csv` or `stats_processed_all_bouts.csv`.
4. From `raw_fighter_details.csv`, only identity, height, weight, reach, stance, and DOB
   are safe. Its career-average columns reflect scrape-time values and leak future results.
5. Preserve the fixed fighter-order symmetry operation when rebuilding training rows.

The leakage guard in `tests/test_features.py` must remain green.

## Repository layout

```text
src/ufcpred/
  db.py          SQLite connection and core schema
  ingest.py      UFC-DataLab CSV loading
  parse.py       raw rows to normalised SQLite tables
  ratings.py     chronological pre-fight Elo
  features.py    point-in-time differential features
  train.py       chronological training and reporting
  predict.py     as-of-date named-matchup prediction CLI
tests/           parser, Elo, leakage, and training helper tests
data/            local database, metadata, warnings, and archives (ignored)
vendor/          local UFC-DataLab checkout (ignored)
models/          trained artifacts and preserved versions (ignored)
reports/         generated evaluation reports (ignored)
```

## Troubleshooting

### `stats_raw.csv not found`

Confirm that `vendor/UFC-DataLab` exists and contains
`data/stats/stats_raw.csv`. Repeat the dataset clone step if this is a fresh local setup.

### `features table not found`

Run the pipeline through parsing, Elo, and feature construction before training:

```powershell
.\.venv\Scripts\python.exe -m ufcpred.parse
.\.venv\Scripts\python.exe -m ufcpred.ratings
.\.venv\Scripts\python.exe -m ufcpred.features
```

### LightGBM installation fails

Confirm that the virtual environment is using a supported 64-bit Python version, then
upgrade pip inside `.venv` and retry the editable install. Do not fall back to a global
package installation.

### Metrics change after a refresh

Check the recorded upstream hash, database counts, maximum event date, ingest warnings,
feature NULL rates, and chronological split sizes. Dataset revisions legitimately change
metrics, but large changes can signal schema drift or leakage.
