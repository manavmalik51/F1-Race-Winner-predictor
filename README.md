# Formula 1 Race Predictor

A season-wide Formula 1 prediction project. One model is trained offline on completed historical races, then reused to predict any upcoming Grand Prix after qualifying. The prediction app does not train a model, and it does not ask users to upload historical CSV files or PDFs.

The app asks the user to choose a Grand Prix from the upcoming calendar. Once the qualifying session is complete and published by the timing source, the app fetches the event's driver entries, qualifying laps, speed-trap readings, and qualifying-session weather. It then displays win probabilities for the full field and a ranked top-three podium.

## Project at a glance

```text
Public race history ──> offline collector ──> feature engineering ──> grouped CV model comparison
                                                                          │
                                                                          v
                                                             selected fitted model bundle
                                                                          │
Upcoming calendar ──> qualifying completed? ──> fetch event data ──> Streamlit prediction
```

The current trained version uses 90 completed races from the 2023, 2024, 2025, and completed 2026 seasons available at the training cutoff. The artifact was selected with race-grouped cross-validation and is saved as `f1_all_races_model.pkl`.

## What is in the project

| File | Purpose |
|---|---|
| `f1_race_pipeline.py` | Data loaders, feature engineering, model comparison and fitting, calendar lookup, automatic qualifying fetch, and inference functions. |
| `fetch_public_f1.py` | Downloads and caches public race history. It writes intermediate data below `work/`; it does not ask for a Kaggle account or local CSV collection. |
| `train_model.py` | Offline training entry point. It engineers features, compares candidate algorithms with `GroupKFold`, fits the selected model, and writes the model bundle and CV report. |
| `streamlit_app.py` | Inference-only interface. It loads the trained bundle, shows upcoming events, fetches qualifying data for the selected event, and displays predictions. |
| `f1_all_races_model.pkl` | Trained model bundle containing the fitted win and podium classifiers, feature metadata, validation scores, and the historical rows needed to calculate prior form. |
| `model_comparison.csv` | Cross-validation comparison for LightGBM, XGBoost, and ExtraTrees. |
| `f1_race_predictor_colab.ipynb` | Colab workflow for running the offline collector and trainer without uploading race data. |
| `requirements.txt` | Python packages and versions used by the project. |
| `README.md` | This guide. |

The project also retains loader support for FastF1 and standard Kaggle/Ergast CSVs. The normal app workflow uses automatic public-data ingestion instead.

## Quick start

### 1. Install dependencies

Run these commands from the folder containing the `outputs` directory:

```bash
python -m pip install -r outputs/requirements.txt
```

The project uses Python with `pandas`, `numpy`, `scikit-learn`, LightGBM, XGBoost, FastF1, and Streamlit. The two non-LightGBM/XGBoost fallbacks and FastF1 support are documented below.

### 2. Start the prediction app

```bash
streamlit run outputs/streamlit_app.py
```

The app loads `outputs/f1_all_races_model.pkl`. It does not run cross-validation or fit a model at launch.

### 3. Predict an upcoming Grand Prix

1. Choose an upcoming race from the calendar list.
2. If its qualifying session is still in the future, the app marks qualifying as pending and disables prediction.
3. Once qualifying data is published, refresh the race list if necessary and click **Fetch qualifying and predict podium**.
4. Review the P1–P3 predictions, the full-field probabilities, and download the prediction CSV if useful.

There is no historical CSV or PDF upload in this workflow. The only routine interaction is selecting an event and requesting a prediction after qualifying.

## Offline training and model refresh

Training is separate from Streamlit. Run it when preparing a new model version or refreshing the historical data:

```bash
python outputs/train_model.py --refresh
```

The trainer:

1. Runs `fetch_public_f1.py` to retrieve completed race history and qualifying records.
2. Builds one driver-race row per entry and engineers the race-safe features described below.
3. Compares three algorithms using the same race-grouped CV folds.
4. Selects the algorithm with the lowest mean race-level winner log loss.
5. Fits separate win and podium classifiers using that selected algorithm and the full eligible history.
6. Saves `outputs/f1_all_races_model.pkl` and `outputs/model_comparison.csv`.

To use an existing local Kaggle/Ergast CSV directory for training instead of the automatic data collector:

```bash
python outputs/train_model.py --csv-dir /path/to/f1-csvs
```

The required CSVs are `results.csv`, `races.csv`, and `qualifying.csv`. `drivers.csv`, `constructors.csv`, and `circuits.csv` are optional lookup tables. The normal automatic workflow avoids this manual file step.

`--folds` can change the number of GroupKFold splits:

```bash
python outputs/train_model.py --refresh --folds 5
```

The public data collector caches JSON responses in `work/public_api_cache/` and writes its standardized source table to `work/f1_history_source.csv`. The collector includes request delays and retries for rate limits. If a provider is temporarily unavailable, retrying can reuse cached responses.

## Data sources and what they provide

The automatic collector combines public sources so the user does not have to assemble data by hand.

### Jolpica (Ergast-compatible API)

The collector uses Jolpica for:

- season calendars and race metadata;
- final race classifications, driver/team names, grid position, points, and status;
- qualifying positions and Q1/Q2/Q3 lap times;
- lap-one timings used to calculate qualifying-to-lap-one deltas.

Races receive a stable internal ID of the form `YEAR-ROUND`, such as `2026-21`. This ID groups every driver row from the same Grand Prix for cross-validation and feature construction.

### OpenF1

OpenF1 provides historical and current session timing data used for:

- qualifying-session and race-session discovery;
- qualifying lap durations and speed-trap (`st_speed`) readings;
- driver acronyms and team names;
- qualifying-session track-temperature readings;
- race lap timing for the 2025 Las Vegas disqualification pace correction.

At inference time, the chosen event is matched to its qualifying session. The app waits until the session end time has passed before marking the event as prediction-ready, then fetches the completed session's timing data.

### Public weather table

A public FastF1-derived weather summary supplies track temperatures for its covered seasons. Las Vegas 2023 and 2024 are backfilled from OpenF1 race-session weather. Track-temperature coverage is incomplete outside those ranges; missing historical values are imputed by the model's training preprocessing.

### FastF1 and Kaggle alternatives

`load_fastf1(...)` is provided for users who want to load sessions directly through the FastF1 Python package. It can read race and qualifying sessions, laps, weather, and qualifying telemetry, depending on source availability and cache setup.

`load_kaggle_csvs(...)` accepts a directory or mapping of standard CSV filenames. It joins records by `raceId` and `driverId`, and can use the optional lookup files to attach readable driver, team, and circuit names. The standard CSVs do not contain all weather or telemetry values, so features unavailable in those files remain missing until enriched.

## Input rows and prediction data

### Historical model rows

The training table contains one row per driver per completed race. Important raw columns include:

| Column | Meaning |
|---|---|
| `race_id` | Stable event grouping key, `YEAR-ROUND`. |
| `year`, `date`, `event_name`, `circuit_name` | Calendar context. |
| `driver`, `team` | Driver code and constructor/team. |
| `grid_position` | Starting-grid position. |
| `qualifying_time_seconds` | Best available qualifying lap time converted to seconds. |
| `finish_position`, `status`, `points` | Published historical race outcome and championship points. |
| `track_temp_c` | Session track temperature when available. |
| `straight_line_speed_kph` | Maximum recorded qualifying speed-trap speed when available. |
| `quali_lap1_delta_seconds` | Race lap-one time minus the qualifying best lap time. Used as prior team history only. |
| `race_pace_seconds` / `pre_dsq_pace_rank` | Optional race-pace evidence for the 2025 Las Vegas DSQ correction. |

The collector currently builds this table automatically. Users generally do not need to view or edit it.

### Qualifying rows at prediction time

The app fetches the qualifying records automatically after the user selects a ready event. Each entry contains the driver/team, qualifying order, best qualifying lap duration, speed-trap speed, event date, event ID, and available track temperature. No finishing position or current-race points are added because those outcomes are unknown before the race.

The starting-grid position is initialized from qualifying lap order. Grid penalties can change the official grid after qualifying; the app flags this because the public session data used here does not guarantee an official penalty-adjusted grid for every event. Update the entry grid before trusting a prediction if penalties change the order.

## Feature engineering

Every model input is constructed so that a race does not see its own finish, points, or lap-one outcome.

### Low-drag circuit cluster

`is_low_drag_cluster` is 1 for Las Vegas, Monza, Baku, and Jeddah. Grouping these similar high-speed circuits gives the model shared examples for straight-line speed despite few Las Vegas editions.

### Straight-line speed rank

Within each race, available qualifying speed-trap speeds are ranked from fastest to slowest as `straight_line_speed_rank`. This is based on speed-trap records when present. It is missing for races without the relevant telemetry and is median-imputed by the model pipeline.

`speed_overtake_strength` combines the speed rank with prior average positions gained. It supplements the speed and grid inputs with a direct racecraft signal. These features do not force a particular finish; the tree models learn their contribution from the training data.

### Prior positions gained and rolling driver form

Historical positions gained are defined as:

```text
positions gained = grid position - adjusted finish position
```

`positions_gained_avg` is the driver's rolling average over up to five **previous** races. `rolling_5_race_points` is the prior-five-race mean of championship points. `rolling_5_race_pace_delta` is the prior-five-race mean of positions gained. All rolling calculations shift by one race before aggregation, so the current event's result is excluded.

`driver_experience_races` counts the driver's earlier race rows, not the current result.

### Cold-track warm-up index

The per-race qualifying-to-lap-one pace delta is:

```text
lap-one delta = lap-one time - best qualifying time
```

For a prediction, the pipeline uses the team's rolling mean delta from up to five earlier races and multiplies it by the current event's track temperature:

```text
cold_track_warmup_index = current track temperature × prior team lap-one delta
```

The current race's lap one is never used. At prediction time the app uses track temperature measured during qualifying as the available race-night proxy. Weather can change between qualifying and the race, so treat it as an estimate.

### 2025 Las Vegas McLaren DSQ handling

The 2025 Las Vegas post-race disqualifications for Norris and Piastri must not be treated as ordinary last-place finishes. Their training targets are replaced with race-pace ranks derived from median race lap durations. When the pace rank cannot be obtained, a DSQ row is excluded from fitting instead of being assigned a false finishing position.

### 2026 regulation-era weighting

`sample_weight` applies a year-based decay of `0.97 ** age_in_years` and multiplies 2026 entries by 3.0. This gives 2026 regulation-era races more influence while retaining earlier seasons for sample size. Weights are passed into each estimator during CV and final fitting.

### Exact model input columns

These are the numeric columns passed into each estimator after feature engineering:

| Feature | Model interpretation |
|---|---|
| `grid_position` | Driver's race starting position (from qualifying order in the automatic feed). |
| `qualifying_time_seconds` | Driver's best qualifying lap time in seconds. |
| `straight_line_speed_rank` | Rank of available qualifying speed-trap speed within the event; 1 is fastest. |
| `positions_gained_avg` | Mean positions gained over up to five prior races. |
| `rolling_5_race_points` | Mean championship points over up to five prior races. |
| `rolling_5_race_pace_delta` | Mean grid-to-finish positions gained over up to five prior races. |
| `track_temp_c` | Track temperature measured during qualifying, when available. |
| `is_low_drag_cluster` | Indicator for Las Vegas, Monza, Baku, or Jeddah. |
| `cold_track_warmup_index` | Track temperature interacted with prior team qualifying-to-lap-one pace delta. |
| `driver_experience_races` | Number of previous race rows for the driver in the loaded history. |
| `speed_overtake_strength` | Combined speed-rank and historical positions-gained signal. |

The classifier does **not** receive the driver's name, team name, event name, `race_id`, current-race points, current-race finish, or lap-one time as numeric predictors. Names and IDs are used for joining, grouping, sorting, and lagged-history calculations. A median imputer fitted inside each CV fold handles missing numeric telemetry values and adds indicators for missingness.

The win target is 1 when the adjusted finish rank is first. The podium target is 1 when the adjusted finish rank is in the top three. Both targets are formed from historical results only; prediction rows have no target result.

## Model comparison, validation, and selection

The project compares three classifiers:

- **LightGBM** — efficient gradient-boosted trees for tabular data.
- **XGBoost** — regularized gradient-boosted trees.
- **ExtraTrees** — randomized tree ensemble baseline.

All candidates use identical `GroupKFold` splits with `race_id` as the group. No race is split across training and validation rows, preventing drivers from the same session/race appearing on both sides of a fold.

Each fold fits two binary classifiers: one for race wins and one for podium finishes. Cross-validation reports:

- `mean_winner_log_loss`: negative log probability assigned to the actual winner after within-race normalization; **lower is better and this selects the model**.
- `winner_top1_accuracy`: fraction of validation races where the highest win probability belongs to the winner.
- `podium_top3_recall`: average fraction of the actual podium drivers included in the model's top-three podium ranking.
- `exact_podium_accuracy`: fraction of races where the predicted top-three set matches the actual top-three set.
- `podium_brier_score`: squared error of the constrained per-driver podium probabilities; lower is better.

For the current 90-race history, the recorded five-fold comparison selected ExtraTrees:

| Algorithm | Mean winner log loss | Winner top-one accuracy | Podium top-three recall | Exact podium accuracy | Podium Brier |
|---|---:|---:|---:|---:|---:|
| ExtraTrees | 1.1348 | 0.6667 | 0.6704 | 0.2333 | 0.0679 |
| XGBoost | 1.1758 | 0.6111 | 0.6852 | 0.2556 | 0.0641 |
| LightGBM | 1.3581 | 0.6111 | 0.6593 | 0.2222 | 0.0704 |

ExtraTrees is selected because winner log loss is the primary criterion. XGBoost scored slightly better on podium recall and Brier score; these are shown so the user can see the trade-off. The full precision results are stored in `model_comparison.csv`.

This evaluation is grouped by race but is not a strict forward-in-time backtest: random GroupKFold may train on some later races while validating on earlier races. For a formal deployment study, add a chronological season/round holdout in addition to GroupKFold.

## Probability outputs and podium ranking

The win model produces one score per driver. The app converts scores to probabilities with a softmax **within the selected race**, so the win probabilities sum to 1 (100%).

The podium classifier produces per-driver scores. A logistic offset is applied within the race so the marginal podium probabilities sum to 3 expected places, while each individual probability stays between 0 and 1. The displayed podium is the three drivers with the highest podium marginal probabilities; win probability breaks ties.

Podium probabilities are marginal chances and do not form a complete joint model of finishing order. The model does not calculate the probability of every possible ordered podium permutation.

## Streamlit interface

The app is intentionally an inference front end:

1. It loads `f1_all_races_model.pkl` from the same directory.
2. It requests upcoming race schedules from Jolpica and qualifying-session schedules from OpenF1.
3. It lists upcoming events and marks qualifying as pending or available.
4. The prediction control remains disabled until the qualifying session end time has passed.
5. It fetches the selected session's lap records, driver list, speed traps, and weather.
6. It builds shifted historical features from the history embedded in the model bundle.
7. It displays P1–P3, all-driver probabilities, probability totals, and a downloadable prediction CSV.

The Streamlit code never calls the training functions. A model refresh must be completed offline, then the app restarted so it loads the new artifact.

## Model bundle contents

`f1_all_races_model.pkl` is a Python pickle containing:

- the selected win classifier;
- the selected podium classifier;
- selected estimator name and feature list;
- CV comparison scores and training metadata;
- raw historical driver-race rows used to build prior rolling features.

The bundle is an executable Python serialization format. Only load a bundle from this project or another trusted source. A bundle trained with different Python/scikit-learn/estimator versions may not be compatible with an older runtime.

## Colab workflow

`f1_race_predictor_colab.ipynb` runs the same offline process in Google Colab. It asks for the small project source files only (`f1_race_pipeline.py`, `fetch_public_f1.py`, and `train_model.py`); the data collector downloads race history itself. The notebook downloads the resulting model PKL and comparison CSV. It is not necessary to upload the race CSVs to Colab.

## Troubleshooting

### Upcoming race is listed, but prediction is disabled

The schedule can list a qualifying session before it has taken place. The app enables prediction only after the session's end time. Refresh the race list after the session completes.

### Qualifying is complete, but timing laps are unavailable

Some data providers publish session metadata before publishing all lap records. Wait for timing data to populate and refresh. The app reports an error rather than generating a prediction from empty qualifying data.

### Calendar or data fetch fails

The calendar, qualifying, and weather fetches require an internet connection and public provider availability. Retry later if an API is rate-limiting. Offline training responses are cached in `work/public_api_cache/`.

### The app says the model file is missing

Run `python outputs/train_model.py --refresh` to produce `outputs/f1_all_races_model.pkl`, then restart Streamlit.

### Probability columns do not sum to one / three

Win probabilities should sum to 1 per event and podium marginals to 3 (subject to floating-point rounding). These are normalized within one race; do not combine rows from multiple race predictions when checking totals.

## Limitations and interpretation

- The 2026 regulation era has a smaller sample than earlier seasons; 3× weighting increases its influence but does not remove uncertainty.
- The high-speed speed-trap feature is populated only where qualifying telemetry is available; missing historical speed is imputed.
- Weather coverage is incomplete in older seasons. The event's qualifying track temperature is a proxy for race conditions.
- Qualifying order is used to initialize the grid. Penalties can change the official starting order.
- The model predicts probabilities, not guarantees. Safety cars, incidents, reliability, strategy, penalties, and changing weather remain major sources of uncertainty.
- The selection metric is winner log loss, not podium recall. The selected model may not be the top podium classifier by every secondary metric.
- Public APIs can change fields, limits, or availability. Update the collector and test the data schema if a provider changes its API.

## Public data references

- [Jolpica F1 API](https://api.jolpi.ca/ergast/f1/)
- [OpenF1 API documentation](https://openf1.org/docs/)
- [FastF1 documentation](https://docs.fastf1.dev/)
- [Public FastF1-derived F1 dataset used for weather summaries](https://github.com/Arths17/f1-dataset)
- [Official Formula 1 calendar](https://www.formula1.com/en/racing/2026)
