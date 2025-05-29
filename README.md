# Formula 1 Top-3 Predictor

## Project overview
This project forecasts Formula 1 race results. It collects historical race data, engineers
leakage-safe features and trains an XGBoost model using time-series cross
validation. Predictions for the upcoming Grand Prix are made via a simple CLI.

High level architecture:
- **DataLoader** – handles downloads with caching.
- **dataset_builder** – aggregates race results into a single dataset.
- **feature_engineering** – converts the dataset into a machine learning matrix.
- **model_training** – runs Optuna search and saves trained models.
- **predict_next_race** – produces the final ranking for the next event.

## Installation & setup
1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost optuna fastf1 requests joblib
   ```
3. (Optional) create a `.env` file to override cache locations or FastF1 settings.
   By default data is cached under `.cache/` and raw files in `data/raw/`.

## Data workflow
`dataset_builder.py` orchestrates data collection. It iterates over seasons,
fetching each race with `DataLoader` and saving the merged result at
`data/processed/f1_dataset.parquet`【F:f1_predictor/dataset_builder.py†L16-L68】.
The builder now loads this file if it exists and skips already processed races,
allowing interrupted runs to resume where they left off.
Run it with:
```bash
python -m f1_predictor.dataset_builder --seasons 2019 2020 2021 --max-attempts 3
```
Without arguments it pulls all seasons from 2018 onwards.

## Feature pipeline
`generate_feature_matrix` converts raw data into training features. It converts
time columns to seconds, builds exponentially weighted rolling metrics using a
`shift(1)` to prevent future leakage and stores the preprocessing pipeline at
`models/feature_pipeline.joblib`【F:f1_predictor/feature_engineering.py†L21-L136】.
Numeric features are scaled; categorical ones are one-hot encoded.

## Model training & evaluation
`model_training.py` performs a 5-fold `TimeSeriesSplit` with Optuna hyperparameter
search before fitting the final model. Each run saves an XGBoost booster to a
`models/<DATE>/model.xgb` directory and logs MAE for every split
【F:f1_predictor/model_training.py†L23-L96】.
Execute training with:
```bash
python -m f1_predictor.model_training --data data/processed/f1_dataset.parquet
```

## Making predictions
`predict_next_race.py` loads the newest model and appends placeholder rows for
the forthcoming event. After feature generation it outputs predicted finishing
positions and the softmax-derived probability of being in the top three
【F:f1_predictor/predict_next_race.py†L142-L178】.
To forecast the next scheduled race simply run:
```bash
python -m f1_predictor.predict_next_race
```
For a specific round provide a slug, e.g.:
```bash
python -m f1_predictor.predict_next_race --race 2025-Canada
```

## CLI cheat-sheet
| Command | Purpose | Key options |
|---------|---------|-------------|
| `python -m f1_predictor.dataset_builder` | Build dataset | `--seasons 2019 2020`, `--max-attempts 3` |
| `python -m f1_predictor.model_training`  | Train model   | `--data <path>` |
| `python -m f1_predictor.predict_next_race` | Predict upcoming GP | `--race YEAR-Slug` |

## Directory structure
After running the whole pipeline the folder tree will resemble:
```
.
├── data
│   ├── processed
│   │   └── f1_dataset.parquet
│   └── raw
│       ├── ergast/
│       └── racefans/
├── models
│   ├── feature_pipeline.joblib
│   └── <DATE>/
│       └── model.xgb
└── logs/
```

## Tuning tips
- Modify the search space inside `train_model` and adjust `study.optimize` for
  more or fewer trials.
- Remove the existing `models/` subfolders before re-running to avoid confusion.

## Extensibility
1. Add new columns to the dataset in `dataset_builder` then update
   `generate_feature_matrix` to include them in `numeric_features` or
   `categorical_features`.
2. Data from other APIs can be fetched by extending `DataLoader` with new
   methods that follow the retry and caching patterns used in existing ones.

### END OF README
### END OF TASK
