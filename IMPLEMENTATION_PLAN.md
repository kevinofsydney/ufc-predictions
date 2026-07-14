# UFC Fighter Performance Prediction System — Implementation Plan

**Audience:** This plan is written for a coding agent (of any vendor) to execute step by step.
Follow the phases in order. Do not skip acceptance checks. Do not invent alternative
designs — all design decisions are already made and pinned in Part 2 below.

This document has two parts:

- **Part 1 — Execution protocol:** how to run the build cost-efficiently across a tiered
  set of models, how phases gate on acceptance checks, and how to record progress so a
  *different* agent can resume mid-build.
- **Part 2 — Pinned technical specification:** the complete, authoritative spec for every
  phase. An executing agent implements Part 2 exactly, in order, using the protocol in
  Part 1.

---

# Part 1 — Execution protocol

## 1.1 Progress checklist (update and commit as each gate passes)

The executing agent MUST tick a box only after that phase's acceptance checks (defined in
Part 2) have passed, and MUST commit the updated checklist in the same commit as the
phase's code. This checklist is the resume point for any future agent.

- [x] Step 0 — This plan committed to the repo
- [x] Phase 0 — Scaffold + DB schema (gate: `init_db()` creates 4 tables)
- [x] Phase 1 — UFC-DataLab ingest (gate: ~8,700 fights / ~3,700 fighters, recent max date)
- [x] Phase 2 — Normalise into SQLite (gate: row counts, >95% non-NULL winner, `pytest tests/test_parse.py`)
- [ ] Phase 3 — Elo engine (gate: unit tests, top-15 sanity, higher-rated wins 62–68%)
- [ ] Phase 4 — Point-in-time features (gate: leakage-guard test, label mean 0.45–0.55, NULL audit)
- [ ] Phase 5 — Training + evaluation (gate: artifacts produced, metrics within sanity bounds)
- [ ] Phase 6 — Prediction CLI (gate: lopsided matchup sanity, probs sum to 100%)
- [ ] Phase 7 — Incremental update (gate: second consecutive run ingests 0 new fights)
- [ ] Wrap-up — README + Definition of Done (Part 2 §11)

## 1.2 Model tiers (cost-efficient allocation)

Work is allocated to three capability tiers. Named models are the current mappings for
the Claude and OpenAI families; any vendor's equivalent-capability model may be
substituted tier-for-tier.

| Tier | Claude | OpenAI | Use for |
|---|---|---|---|
| **T1 — workhorse** | Sonnet | Luna | Mechanical, fully-specified work |
| **T2 — heavy** | Opus | Terra | Messy real-world data, judgment-heavy implementation |
| **T3 — frontier** | Fable | Sol | Leakage-critical logic, orchestration, auditing every phase gate |

The T3 model is the **orchestrator**: it runs as the main thread/session, dispatches each
phase to its assigned tier as a subagent (or performs it itself where marked T3), and
personally runs every phase's acceptance checks before advancing. T1/T2 tokens are cheap;
T3 tokens are spent only on prompts, gate audits, and the phases explicitly assigned to it.

## 1.3 Phase → tier allocation

| Phase | What | Tier | Why |
|---|---|---|---|
| 0 | Scaffold + schema (verbatim from spec) | **T1** | Mechanical file creation |
| 1 | Clone UFC-DataLab, `ingest.py` `load_raw()` | **T1** | CSV loading with pinned separators |
| 2 | Normalise → SQLite (5 converters, surrogate keys, tests) | **T2** | Messy real-world data, many edge cases |
| 3 | Elo engine + tests | **T1** | Algorithm is spelled out verbatim in the spec |
| 4 | Point-in-time feature builder + leakage-guard test | **T3** | Leakage-critical — "the most common bug in this kind of system" |
| 5 | Training + evaluation + calibration | **T2** implements; **T3** audits metrics | Boilerplate sklearn/LightGBM, but result interpretation needs judgment |
| 6 | Prediction CLI | **T1** | Straightforward glue over existing pieces |
| 7 | Incremental update script | **T1** | Wraps the vendored Scrapy spider + re-ingest |
| — | README + final end-to-end check | **T1** writes README; **T3** runs Definition of Done | |

## 1.4 Dispatch and gate rules

1. **Strictly sequential.** Phase N+1 does not start until phase N's gate passes.
2. **Self-contained prompts.** When dispatching a phase to T1/T2, the orchestrator
   includes: (a) the verbatim Part 2 text for that phase, (b) the global rules
   (Part 2 §0), (c) the environment adaptations (§1.5 below), and (d) any facts learned
   in earlier phases that the phase needs (e.g. exact CSV column headers from Phase 1
   feed the Phase 2 prompt). The subagent must not need to re-derive context.
3. **Gates are run by T3**, not trusted from a subagent's self-report.
4. **Two-strikes rule.** If a phase's output fails its gate twice after subagent retries,
   the T3 orchestrator fixes it directly instead of re-dispatching.
5. **Commit per phase.** One commit at each passing gate (code + updated §1.1 checklist).
6. **Deviations are recorded.** Any workaround or departure from Part 2 is appended to
   §1.6 in the same commit that introduces it.

## 1.5 Environment adaptations (this repo)

These adapt Part 2 to the actual repo/host without changing any design decision:

- **Project lives at the repo root** (`ufc-predictions/`), not in a nested
  `ufc-predictor/` folder. The Part 2 §1 layout applies with its top-level folder
  stripped: `pyproject.toml`, `src/ufcpred/`, `tests/`, `data/` sit at the root.
- **Windows host:** verify DB contents via Python's `sqlite3` module (the `sqlite3` CLI
  may be absent); write the Phase 7 refreshed CSV to a local temp/scratch directory, not
  `/tmp`; use a `.venv` virtualenv; `.gitignore` includes `data/`, `vendor/`, `.venv/`,
  `models/`.
- **Dependencies:** the Part 2 §1 list plus `joblib` (model persistence) and `scrapy`
  (Phase 7). Pin versions only if wheels fail to install for the host's Python.
- **Phase 7 fallback scope:** if the vendored Scrapy spider fails on this host, log the
  failure, note the Appendix A fallback in the README, and continue — implementing
  Appendix A is out of v1 scope.

## 1.6 Deviations log

- **Phase 2:** Real data differs from spec assumptions, handled without design change:
  `event_date` is `DD/MM/YYYY` (not month-name); `"---"` appears as a null sentinel;
  `method` also contains `TKO - Doctor's Stoppage` (mapped → `KO/TKO`). The
  Sakuraba–Silveira same-night rematch at UFC Ultimate Japan collides under the
  name-based fight key (1 fight lost of 8,737 — accepted per the known-limitation rule,
  logged in `data/ingest_warnings.log`). Fighters from the details file who never
  fought are also inserted (4,110 total) so the predict CLI can resolve any known name.

---

# Part 2 — Pinned technical specification

**Follow the phases in order. Do not skip acceptance checks. Do not invent alternative
designs — all design decisions are already made and pinned below.**

## 0. Pinned decisions (do not change these)

| Decision | Value |
|---|---|
| Language | Python 3.11+ |
| Database | SQLite, single file at `data/ufc.db` |
| HTTP | `requests` with retries, 2-second delay between requests |
| Parsing | `beautifulsoup4` + `lxml` |
| Data source | UFC-DataLab repo CSVs (github.com/komaksym/UFC-DataLab) for bootstrap; its bundled Scrapy spider for updates. Fall back to a custom scraper (Appendix A) only if the repo dies. |
| Rating system | Elo with method-of-victory K scaling (v1); Glicko-2 optional in v2 |
| Model | Logistic regression baseline, then LightGBM |
| Validation split | Chronological: train < 2023-01-01, validation 2023, test 2024+ |
| Metrics | Log loss (primary), Brier score, accuracy, calibration plot |
| Project layout | See section 1 |

**Global rules that apply to every phase:**

1. **Point-in-time rule:** Any feature describing a fighter before fight F must be
   computed ONLY from fights whose date is strictly earlier than F's date. Never include
   fight F itself or later fights. This is the most common bug in this kind of system —
   check it constantly.
2. **Never re-scrape unnecessarily:** Every downloaded HTML page is saved to
   `data/raw_html/` keyed by URL hash. The parser reads from disk, never from the
   network. Only the scraper touches the network.
3. **Politeness:** One request at a time, `time.sleep(2)` between requests, User-Agent
   set to a descriptive string. If a request fails, retry up to 3 times with exponential
   backoff, then log and continue.
4. **Idempotency:** Every pipeline step can be re-run safely. Use `INSERT OR REPLACE` /
   upserts keyed on stable IDs.
5. **IDs:** ufcstats.com URLs contain stable hex IDs (e.g.
   `.../fighter-details/029eba4f2b3daf13`). Use that hex string as the primary key for
   fighters, fights, and events.

## 1. Project layout

Create exactly this structure (per §1.5, the top-level folder is the repo root):

```
ufc-predictor/
├── pyproject.toml            # deps: requests, beautifulsoup4, lxml, pandas,
│                             #       numpy, scikit-learn, lightgbm, matplotlib, pytest
├── data/
│   ├── raw_html/             # cached pages (gitignored)
│   └── ufc.db                # SQLite database (gitignored)
├── src/ufcpred/
│   ├── __init__.py
│   ├── db.py                 # connection helper + schema creation
│   ├── scrape.py             # download pages into raw_html cache
│   ├── parse.py              # raw_html -> SQLite tables
│   ├── ratings.py            # Elo engine
│   ├── features.py           # point-in-time feature builder
│   ├── train.py              # model training + evaluation
│   ├── predict.py            # CLI: predict an upcoming fight
│   └── update.py             # weekly incremental update
└── tests/
    ├── test_parse.py
    ├── test_ratings.py
    └── test_features.py
```

## 2. Phase 0 — Bootstrap (do this first)

**Goal:** Working skeleton with database schema.

Tasks:

1. Create the project layout above. Initialise git. Add `data/` to `.gitignore`.
2. In `db.py`, write `get_conn()` returning a SQLite connection with
   `PRAGMA foreign_keys=ON`, and `init_db()` that executes the schema below.

**Schema (create verbatim):**

```sql
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,      -- hex id from URL
    name        TEXT NOT NULL,
    event_date  TEXT NOT NULL,         -- ISO 8601 'YYYY-MM-DD'
    location    TEXT
);
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id  TEXT PRIMARY KEY,      -- hex id from URL
    name        TEXT NOT NULL,
    height_cm   REAL,                  -- NULL if unknown
    reach_cm    REAL,
    stance      TEXT,                  -- 'Orthodox','Southpaw','Switch', or NULL
    dob         TEXT                   -- ISO 8601 or NULL
);
CREATE TABLE IF NOT EXISTS fights (
    fight_id     TEXT PRIMARY KEY,     -- hex id from URL
    event_id     TEXT NOT NULL REFERENCES events(event_id),
    fighter_a_id TEXT NOT NULL REFERENCES fighters(fighter_id),
    fighter_b_id TEXT NOT NULL REFERENCES fighters(fighter_id),
    winner_id    TEXT,                 -- NULL for draw/no-contest
    weight_class TEXT,
    method       TEXT,                 -- e.g. 'KO/TKO','SUB','U-DEC','S-DEC','M-DEC','DQ','NC'
    end_round    INTEGER,
    end_time_sec INTEGER,              -- seconds into end_round
    is_title     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS fight_stats (
    fight_id     TEXT NOT NULL REFERENCES fights(fight_id),
    fighter_id   TEXT NOT NULL REFERENCES fighters(fighter_id),
    knockdowns   INTEGER,
    sig_strikes_landed    INTEGER,
    sig_strikes_attempted INTEGER,
    total_strikes_landed  INTEGER,
    total_strikes_attempted INTEGER,
    takedowns_landed      INTEGER,
    takedowns_attempted   INTEGER,
    sub_attempts INTEGER,
    reversals    INTEGER,
    control_time_sec INTEGER,
    PRIMARY KEY (fight_id, fighter_id)
);
```

**Acceptance check:** `python -c "from ufcpred.db import init_db; init_db()"` creates
`data/ufc.db` containing 4 tables. Verify the table list (via the Python `sqlite3`
module on hosts without the `sqlite3` CLI).

## 3. Phase 1 — Ingest UFC-DataLab (replaces custom scraping)

**Goal:** Load the UFC-DataLab datasets instead of scraping ufcstats.com ourselves.

Background: `https://github.com/komaksym/UFC-DataLab` contains a maintained, pre-scraped
dump of every UFC fight (1994 → present, ~8,700 bouts, updated roughly quarterly) plus
fighter details and OCR'd judge scorecards. It also ships the Scrapy spiders used to
regenerate the data.

**Files to use:**

- `data/stats/stats_raw.csv` (separator `;`) — one row per fight: red/blue fighter
  names, event name/date/location, result, method, round, time, time_format, referee,
  bonus, and per-fighter fight stats (KD, sig strikes as `"23 of 63"` strings, total
  strikes, TD, sub attempts, reversals, control time as `"3:08"`, plus head/body/leg and
  distance/clinch/ground breakdowns).
- `data/external_data/raw_fighter_details.csv` (separator `,`) — one row per fighter:
  Height, Weight, Reach, Stance, DOB, and career-average stats.
- `data/scorecards/OCR_parsed_scorecards/SCORECARDS.csv` — judge-by-judge round scores
  (optional, v2 feature source).

**Files you must NOT use for modelling:**

- `stats_processed.csv` and `stats_processed_all_bouts.csv` contain `*_cs`
  career-snapshot columns (e.g. `red_fighter_slpm_cs`). These are the fighter's CURRENT
  career averages stamped onto every historical row — the same value appears on a 2012
  fight and a 2026 fight. Using them as features is data leakage. Ignore these files
  entirely; we compute point-in-time stats ourselves in Phase 4. The same applies to the
  career-average columns (SLpM, Str_Acc, …) in `raw_fighter_details.csv` — load only
  Height, Weight, Reach, Stance, DOB from it.

Tasks:

1. `git clone --depth 1 https://github.com/komaksym/UFC-DataLab.git vendor/UFC-DataLab`
   (add `vendor/` to `.gitignore`). Record the upstream commit hash in
   `data/ingest_meta.json`.
2. In `scrape.py` (rename it `ingest.py`), write `load_raw() -> (fights_df, fighters_df)`
   reading the two CSVs with the correct separators. Strip whitespace from all string
   columns; uppercase fighter names for joining.

**Acceptance check:** `load_raw()` returns ~8,700 fight rows and ~3,700 fighter rows;
max `event_date` is within the last 4 months.

## 4. Phase 2 — Normalise into SQLite (replaces HTML parsing)

**Goal:** Populate the Phase 0 schema from the two dataframes.

Because the repo has no stable ufcstats hex IDs, generate deterministic surrogate keys:

- `fighter_id = sha1(upper(fighter_name))[:16]`
- `event_id   = sha1(event_name + '|' + event_date)[:16]`
- `fight_id   = sha1(event_id + '|' + red_name + '|' + blue_name)[:16]`

**Known limitation to log, not fix:** keys are name-based, and MMA has genuine duplicate
fighter names (ufcstats itself lists two different "Bruno Silva"s). The repo's fighter
file currently has zero duplicate names, which likely means such fighters were merged
upstream. Accept this for v1; write any join anomalies to `data/ingest_warnings.log`.

Conversions (implement each as a small pure function with a unit test):

1. `"23 of 63"` → `(landed=23, attempted=63)`; `"--"` or empty → `(NULL, NULL)`.
2. Control/end time `"3:08"` → 188 seconds; `"--"` → NULL.
3. Height `5' 11"` → cm; Reach `72"` → cm; `--` → NULL.
4. Date `"July 11, 2026"` or ISO → `YYYY-MM-DD`.
5. Method normalisation: map `"Decision - Unanimous"`→`U-DEC`,
   `"Decision - Split"`→`S-DEC`, `"Decision - Majority"`→`M-DEC`, `"KO/TKO"`→`KO/TKO`,
   `"Submission"`→`SUB`, `"DQ"`→`DQ`,
   `"Overturned"/"Could Not Continue"/"Other"`→`NC`. Rows with `fight_outcome` = draw →
   method `DRAW`, `winner_id` NULL; no-contest → `NC`, `winner_id` NULL.
6. Winner: `fight_outcome` column (`red_win`/`blue_win`/`draw`/`nc`) → `winner_id`.
7. `is_title`: 1 if `bout_type` contains `"Title"`.
8. Map red→fighter_a, blue→fighter_b everywhere (the Phase 4 random swap removes corner
   bias later).

Insert order: fighters, events, fights, fight_stats (two rows per fight). Fighters
appearing in fights but missing from `raw_fighter_details.csv` get a fighters row with
NULL physicals.

Write `tests/test_parse.py` covering the 5 converters plus one full sample row.

**Acceptance checks:**

- `SELECT COUNT(*) FROM fights` ≈ 8,700; `fighters` ≥ 3,700; `fight_stats` ≈ 2× fights.
- \> 95% of fights have non-NULL winner_id.
- Spot-check UFC 329 main card rows against ufcstats.com by hand.
- `pytest tests/test_parse.py` passes.

## 5. Phase 3 — Elo rating engine (`ratings.py`)

**Goal:** For every fight, a pre-fight Elo rating for both fighters, computed
chronologically.

**Exact algorithm (implement verbatim):**

Constants:

```python
START_RATING = 1500.0
BASE_K = 36.0
NEW_FIGHTER_BOOST = 1.5   # applied for a fighter's first 5 UFC fights
METHOD_MULT = {
    "KO/TKO": 1.25, "SUB": 1.25,
    "U-DEC": 1.0, "M-DEC": 0.9, "S-DEC": 0.8,
    "DQ": 0.7, "DRAW": 1.0, "NC": 0.0,   # NC: no rating change
}
```

Procedure:

1. Load all fights joined to event dates, ordered by `event_date ASC, fight_id ASC`.
   This ordering must be deterministic.
2. Maintain `ratings: dict[fighter_id, float]` and `n_fights: dict[fighter_id, int]`,
   both defaulting to `START_RATING` / `0`.
3. For each fight, in order:
   a. Record pre-fight ratings `ra, rb` (these are what get stored — the model must only
      ever see pre-fight values).
   b. `expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))`
   c. Score: winner gets `s_a = 1.0` / loser `0.0`; draw → both `0.5`; NC → skip update
      entirely.
   d. `k = BASE_K * METHOD_MULT[method]`; for each fighter independently, if that
      fighter's `n_fights < 5`, multiply their own k by `NEW_FIGHTER_BOOST`.
   e. `ratings[a] += k_a * (s_a - expected_a)`;
      `ratings[b] += k_b * ((1 - s_a) - (1 - expected_a))`.
   f. Increment both fighters' `n_fights`.
4. Store results in a new table:

```sql
CREATE TABLE IF NOT EXISTS fight_elo (
    fight_id   TEXT PRIMARY KEY REFERENCES fights(fight_id),
    elo_a_pre  REAL NOT NULL,
    elo_b_pre  REAL NOT NULL,
    n_fights_a INTEGER NOT NULL,   -- prior UFC fights, pre-fight
    n_fights_b INTEGER NOT NULL
);
```

Also write `current_ratings()` returning every fighter's latest rating, for inspection.

**Tests (`tests/test_ratings.py`):**

- Two 1500 fighters, A wins by U-DEC with ≥5 prior fights each: A becomes 1518, B
  becomes 1482.
- Expected score of 1600 vs 1400 ≈ 0.7597 (assert to 4 dp).
- An NC changes nothing.

**Acceptance checks:**

- `fight_elo` has one row per fight.
- Sanity: print the top 15 fighters by current rating. Recognisable elite names should
  dominate. If the list looks random, the chronological ordering or update sign is wrong.
- Predictive sanity: over all fights where both fighters had ≥5 prior fights, the
  higher-rated fighter should win ~62–68% of the time. If it's near 50%, there is a bug.

## 6. Phase 4 — Point-in-time feature builder (`features.py`)

**Goal:** One training row per fight with differential features, leakage-free.

For a given fighter as of date D, compute career aggregates over ONLY their fights
strictly before D:

- `slpm` — significant strikes landed per minute (total landed / total fight minutes)
- `sapm` — significant strikes absorbed per minute (opponent's landed / minutes)
- `str_acc` — sig strikes landed / attempted
- `str_def` — 1 − (opponent landed / opponent attempted)
- `td_avg` — takedowns landed per 15 minutes
- `td_acc` — takedowns landed / attempted (NULL if 0 attempts)
- `sub_avg` — submission attempts per 15 minutes
- `ctrl_pct` — control time / total fight time
- `finish_rate` — share of wins by KO/TKO or SUB
- `win_streak` — consecutive wins entering the fight
- `n_prior_fights`, `days_since_last_fight`
- `age_at_fight` — from DOB (NULL if unknown)

Fight minutes for a past fight = `(end_round − 1) * 5 + end_time_sec / 60`.

Then for each fight build the row:

```
fight_id, event_date,
elo_diff        = elo_a_pre − elo_b_pre,
age_diff, reach_diff, height_diff,
slpm_diff, sapm_diff, str_acc_diff, str_def_diff,
td_avg_diff, td_acc_diff, sub_avg_diff, ctrl_pct_diff,
finish_rate_diff, win_streak_diff, n_fights_diff, layoff_diff,
label = 1 if winner is fighter A else 0
```

Rules:

1. Exclude fights with NULL winner (draw/NC) from the training set.
2. **Symmetry / no position leakage:** ufcstats consistently lists the winner in one
   position. To stop the model learning "fighter A wins", randomly swap A and B (and
   negate all diffs, flip the label) with p=0.5 using a fixed seed (42).
3. Missing values: leave as NULL/NaN (LightGBM handles them; for logistic regression
   impute with the training-set median — computed on training rows only).
4. Materialise into a `features` table or a saved parquet file `data/features.parquet`.

**Test (`tests/test_features.py`):** Build a tiny synthetic database of 3 fights across
3 dates for one fighter and assert that the features for fight 3 include exactly fights
1 and 2 — not fight 3. This test is the leakage guard; it must exist.

**Acceptance checks:**

- Row count = number of fights with a winner.
- `label` mean is between 0.45 and 0.55 after the symmetry swap. If it's ~0.65+, the
  swap isn't working.
- No feature column is more than 60% NULL except td_acc/ctrl_pct-type columns for very
  old fights.

## 7. Phase 5 — Model training and evaluation (`train.py`)

**Goal:** Trained, evaluated, calibrated model beating naive baselines.

Tasks:

1. Split chronologically by `event_date`: train `< 2023-01-01`, validation
   `2023-01-01 … 2023-12-31`, test `>= 2024-01-01`. **Never use a random split.**
2. Baselines to report first:
   - Coin flip (log loss 0.693).
   - Elo-only: probability from the Elo formula on `elo_diff` alone.
3. Model 1: `LogisticRegression` (scikit-learn, `C=1.0`, features standardised on
   training statistics, medians imputed from training set).
4. Model 2: LightGBM (`objective='binary'`, `num_leaves=31`, `learning_rate=0.05`,
   `n_estimators=2000`, early stopping on validation log loss, rounds=100).
5. Report for every model on validation AND test: log loss, Brier score, accuracy. Save
   to `reports/metrics.md`.
6. Calibration: plot predicted probability (10 bins) vs actual win rate on the test set
   → `reports/calibration.png`. If the curve deviates badly from the diagonal, wrap the
   model in `CalibratedClassifierCV` (isotonic) fitted on the validation split.
7. Feature importance: save LightGBM gain importances to
   `reports/feature_importance.md`. `elo_diff` should be at or near the top — if it
   isn't, suspect a feature bug.
8. Persist the final model + imputation medians + feature column order to
   `models/model.pkl` with `joblib`.

**Expected results (use as sanity bounds, not goals to force):**

- Elo-only test accuracy ~60–65%, log loss ~0.63–0.66.
- LightGBM should beat Elo-only log loss by a small margin (~0.005–0.02). Small is
  normal.
- If any model's test accuracy exceeds ~72%, assume data leakage and audit Phase 4
  before celebrating.

**Acceptance check:** `python -m ufcpred.train` runs end-to-end and produces
`models/model.pkl`, `reports/metrics.md`, `reports/calibration.png`.

## 8. Phase 6 — Prediction CLI (`predict.py`)

**Goal:** `python -m ufcpred.predict "Fighter One" "Fighter Two" --date 2026-08-01`
prints both win probabilities.

Tasks:

1. Resolve names to fighter_ids (case-insensitive exact match; on ambiguity print
   candidates and exit).
2. Build the same feature vector as Phase 4, as of `--date` (default today), using each
   fighter's full history before that date and their current Elo.
3. Load `models/model.pkl`, predict, print:
   ```
   Fighter One:  63.4%
   Fighter Two:  36.6%
   (elo: 1587 vs 1541 | model: lightgbm | fights in db: 14 vs 9)
   ```
4. If either fighter has < 3 prior UFC fights, append a warning line that the prediction
   is low-confidence.

**Acceptance check:** Predicting a well-known lopsided matchup gives a sensible
favourite; both probabilities sum to 100%.

## 9. Phase 7 — Incremental updates (`update.py`)

**Goal:** Weekly refresh, not dependent on the repo's quarterly cadence.

The repo ships its own Scrapy spider at `src/scraping/ufc_stats/` which regenerates
`stats_raw.csv` from ufcstats.com. Use it rather than writing a new scraper.

Tasks:

1. `update.py` runs, from inside `vendor/UFC-DataLab/src/scraping/ufc_stats/`:
   `python -m scrapy crawl stats_spider -O <temp-dir>/ufc_stats_refresh.csv` (respect
   its settings; do not raise concurrency or remove delays).
2. Diff the refreshed CSV against the `fights` table using the Phase 2 surrogate keys;
   ingest only new rows (and new fighters, via the fighter-details spider or NULL
   physicals).
3. Also `git -C vendor/UFC-DataLab pull` monthly to pick up upstream fixes; re-run full
   ingest when the upstream commit changes (full ingest is idempotent and takes seconds).
4. Recompute the entire `fight_elo` table from scratch after any ingest.
5. Rebuild features for new fights; retrain monthly.
6. Log a summary line: `"update: 13 new fights, 2 new fighters"`.

**Fallback:** if the spider breaks (site layout change) and upstream is unmaintained,
implement Appendix A.

**Acceptance check:** Running twice in a row → second run ingests 0 new fights.

## 10. Phase 8 (optional, v2) — Improvements in priority order

1. **Glicko-2** replacing Elo: adds rating deviation (uncertainty) that grows with
   inactivity. Use the `glicko2` PyPI package; rating period = one calendar month. Feed
   `rd_diff` as an additional feature.
2. **Per-round stats:** parse round-by-round tables for pace/fade features (e.g. round-3
   output vs round-1).
3. **Method prediction:** a second model (multiclass: KO/SUB/DEC) trained the same way.
4. **Odds benchmark:** if a historical odds dataset is available, compare model log loss
   to implied-probability log loss (after removing the vig). Beating closing odds is
   very rare; matching them is a strong result.
5. **Weight-class change and short-notice flags** as features.

## 11. Definition of done (v1)

- [ ] All acceptance checks in phases 0–7 pass.
- [ ] `pytest` passes (parser, ratings, leakage-guard tests).
- [ ] Fresh clone → ingest → parse → ratings → features → train → predict works end to
      end with no manual intervention.
- [ ] Test-set log loss < 0.66 and higher-Elo-wins sanity holds at 62–68%.
- [ ] README documents the exact command sequence.

## Appendix A — Fallback custom scraper (only if UFC-DataLab is abandoned)

ufcstats.com URLs contain unguessable hex IDs, but they are all discoverable by crawling
from one entry point: `http://ufcstats.com/statistics/events/completed?page=all` lists
every event; each event page links its fight-details pages; each fight row links both
fighter-details pages. Crawl breadth-first, cache every page to
`data/raw_html/<sha256(url)>.html`, 2-second delay, then parse from the cache into the
same Phase 0 schema using the ufcstats hex IDs as primary keys. Acceptance: >8,000
cached pages, >7,000 fights parsed.
