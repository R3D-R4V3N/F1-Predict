from __future__ import annotations

"""Predict finish positions for the next Formula 1 race."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import unicodedata

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from .data_loader import DataLoader
from .feature_engineering import MODEL_PATH, _convert_time_to_seconds, _ewm_feature

LOGGER = logging.getLogger(__name__)

# Fallback mapping of race order when the Ergast API does not yet contain a
# schedule for a given season. This only includes a subset of rounds which is
# sufficient for basic usage and unit tests.
FALLBACK_SCHEDULE: dict[int, list[dict[str, object]]] = {
    2025: [
        {"round": 1, "raceName": "Bahrain Grand Prix", "slug": "bahrain"},
        {"round": 2, "raceName": "Saudi Arabian Grand Prix", "slug": "saudiarabia"},
        {"round": 3, "raceName": "Australian Grand Prix", "slug": "australia"},
        {"round": 4, "raceName": "Japanese Grand Prix", "slug": "japan"},
        {"round": 5, "raceName": "Chinese Grand Prix", "slug": "china"},
        {"round": 6, "raceName": "Miami Grand Prix", "slug": "miami"},
        {"round": 7, "raceName": "Emilia Romagna Grand Prix", "slug": "imola"},
        {"round": 8, "raceName": "Monaco Grand Prix", "slug": "monaco"},
        {"round": 9, "raceName": "Canadian Grand Prix", "slug": "canada"},
        {"round": 10, "raceName": "Spanish Grand Prix", "slug": "spain"},
        {"round": 11, "raceName": "Austrian Grand Prix", "slug": "austria"},
        {"round": 12, "raceName": "British Grand Prix", "slug": "britain"},
        {"round": 13, "raceName": "Hungarian Grand Prix", "slug": "hungary"},
        {"round": 14, "raceName": "Belgian Grand Prix", "slug": "belgium"},
        {"round": 15, "raceName": "Dutch Grand Prix", "slug": "netherlands"},
        {"round": 16, "raceName": "Italian Grand Prix", "slug": "italy"},
        {"round": 17, "raceName": "Azerbaijan Grand Prix", "slug": "azerbaijan"},
        {"round": 18, "raceName": "Singapore Grand Prix", "slug": "singapore"},
        {"round": 19, "raceName": "United States Grand Prix", "slug": "usa"},
        {"round": 20, "raceName": "Mexico City Grand Prix", "slug": "mexico"},
        {"round": 21, "raceName": "São Paulo Grand Prix", "slug": "brazil"},
        {"round": 22, "raceName": "Las Vegas Grand Prix", "slug": "lasvegas"},
        {"round": 23, "raceName": "Qatar Grand Prix", "slug": "qatar"},
        {"round": 24, "raceName": "Abu Dhabi Grand Prix", "slug": "abudhabi"},
    ]
}

DATASET_PATH = Path("data/processed/f1_dataset.parquet")
MODELS_DIR = Path("models")


def _latest_model_path() -> Path:
    """Return path to the newest trained model."""
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError("No trained models found in 'models/' directory")
    latest_dir = max(model_dirs, key=lambda p: p.name)
    path = latest_dir / "model.xgb"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path


def _detect_next_race() -> Tuple[int, int]:
    """Detect the next scheduled race via the Ergast API."""
    url = "https://ergast.com/api/f1/current/next.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    races = resp.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        raise RuntimeError("No upcoming race found")
    race = races[0]
    return int(race["season"]), int(race["round"])


def _parse_race_id(race_id: str) -> Tuple[int, int]:
    """Parse a race identifier of the form 'YEAR-SLUG'."""
    try:
        year_part, slug = race_id.split("-", 1)
        year = int(year_part)
    except Exception as exc:  # pragma: no cover - invalid format
        raise ValueError(f"Invalid race id: {race_id}") from exc

    loader = DataLoader()
    schedule = loader.fetch_season(year)
    slug = slug.lower().replace(" ", "")
    for _, row in schedule.iterrows():
        name = str(row.get("raceName") or row.get("RaceName") or "")
        norm = unicodedata.normalize("NFKD", name)
        norm = "".join(c for c in norm if not unicodedata.combining(c))
        norm = norm.lower()
        if slug in norm.replace(" ", ""):
            rnd = int(row.get("round") or row.get("Round"))
            return year, rnd

    # If Ergast doesn't provide the schedule yet, fall back to the
    # built-in mapping for known seasons.
    fallback = FALLBACK_SCHEDULE.get(year)
    if fallback is not None:
        for row in fallback:
            norm = row["slug"]
            if slug == norm:
                return year, int(row["round"])
    raise ValueError(f"Race '{race_id}' not found")


def _prepare_prediction_dataframe(df_hist: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Append placeholder rows for the upcoming race and engineer features."""
    if "driver_id" not in df_hist.columns:
        raise ValueError("Historical dataframe must contain 'driver_id'")

    df_hist = df_hist.copy()
    df_hist.sort_values(["driver_id", "year", "round"], inplace=True)

    latest = df_hist.groupby("driver_id").tail(1).reset_index(drop=True)
    latest["year"] = year
    latest["round"] = rnd
    for col in [
        "Position",
        "GridPosition",
        "FastestLapTime",
        "FastestLapSpeed",
        "racefans_rating",
        "air_temp",
        "humidity",
        "circuit_id",
        "track_length",
        "pit_stops",
        "overtake_difficulty",
    ]:
        if col in latest.columns:
            latest[col] = np.nan
    if "Points" in latest.columns:
        latest["Points"] = 0

    df_all = pd.concat([df_hist, latest], ignore_index=True)

    if "GridPosition" in df_all.columns:
        df_all.rename(columns={"GridPosition": "grid_position"}, inplace=True)
    if "FastestLapTime" in df_all.columns:
        df_all["fastest_lap_time"] = _convert_time_to_seconds(df_all["FastestLapTime"])
    if "FastestLapSpeed" in df_all.columns:
        df_all.rename(columns={"FastestLapSpeed": "fastest_lap_speed"}, inplace=True)

    if "track_length" in df_all.columns:
        df_all["track_length"] = pd.to_numeric(df_all["track_length"], errors="coerce")
    else:
        df_all["track_length"] = np.nan
    df_all["overtake_difficulty"] = 1 / df_all["track_length"]
    if "pit_stops" in df_all.columns:
        df_all["pit_stops"] = pd.to_numeric(df_all["pit_stops"], errors="coerce")
    else:
        df_all["pit_stops"] = np.nan
    if "circuit_id" not in df_all.columns:
        df_all["circuit_id"] = np.nan

    df_all.sort_values(["driver_id", "year", "round"], inplace=True)

    for col in ["Points", "Position", "fastest_lap_time", "fastest_lap_speed", "pit_stops"]:
        if col in df_all.columns:
            feat_name = f"{col.lower()}_ewm"
            df_all[feat_name] = _ewm_feature(df_all, col)

    if "Points" in df_all.columns:
        df_all["points_cumsum"] = df_all.groupby("driver_id")["Points"].cumsum().shift(1)
    else:
        df_all["points_cumsum"] = np.nan

    numeric_features: list[str] = []
    if "grid_position" in df_all.columns:
        numeric_features.append("grid_position")
    if "racefans_rating" in df_all.columns:
        numeric_features.append("racefans_rating")
    if "fastest_lap_time" in df_all.columns:
        numeric_features.append("fastest_lap_time")
    if "fastest_lap_speed" in df_all.columns:
        numeric_features.append("fastest_lap_speed")
    if "track_length" in df_all.columns:
        numeric_features.append("track_length")
        numeric_features.append("overtake_difficulty")
    if "pit_stops" in df_all.columns:
        numeric_features.append("pit_stops")
    if "air_temp" in df_all.columns:
        numeric_features.append("air_temp")
    if "humidity" in df_all.columns:
        numeric_features.append("humidity")
    for col in ["points_cumsum", "points_ewm", "position_ewm", "fastest_lap_time_ewm", "fastest_lap_speed_ewm", "pit_stops_ewm"]:
        if col in df_all.columns:
            numeric_features.append(col)

    numeric_features.extend(["year", "round"])
    categorical_features = ["driver_id"]
    if "TeamName" in df_all.columns:
        categorical_features.append("TeamName")
    if "circuit_id" in df_all.columns:
        categorical_features.append("circuit_id")
    feature_cols = numeric_features + categorical_features

    preprocessor: ColumnTransformer = joblib.load(MODEL_PATH)
    X_all = preprocessor.transform(df_all[feature_cols])
    X_all = X_all.toarray() if hasattr(X_all, "toarray") else X_all
    features_df = pd.DataFrame(
        X_all,
        columns=preprocessor.get_feature_names_out(feature_cols),
        index=df_all.index,
    )

    mask = (df_all["year"] == year) & (df_all["round"] == rnd)
    features_df = features_df[mask]
    drivers = df_all.loc[mask, "driver_id"].reset_index(drop=True)
    features_df["driver"] = drivers
    return features_df


def predict_next(race_id: Optional[str] = None) -> pd.DataFrame:
    """Predict finishing positions for the next race."""
    if race_id is None:
        year, rnd = _detect_next_race()
    else:
        year, rnd = _parse_race_id(race_id)

    model_path = _latest_model_path()
    model = XGBRegressor()
    model.load_model(str(model_path))

    # fetch latest sessions / telemetry / weather for caching side effects
    loader = DataLoader()
    for session in ("FP3", "Q"):
        try:
            loader.fetch_session(year, rnd, session)
        except Exception:  # pragma: no cover - network/availability issues
            LOGGER.info("Could not fetch session %s", session)
    try:
        loader.fetch_session(year, rnd, "R")
    except Exception:  # pragma: no cover - runtime errors
        LOGGER.info("Could not fetch race session")

    df_hist = pd.read_parquet(DATASET_PATH)
    features_df = _prepare_prediction_dataframe(df_hist, year, rnd)
    drivers = features_df.pop("driver")

    preds = model.predict(features_df)
    pred_pos = np.round(preds).astype(int)
    inv_scores = 1 / np.clip(pred_pos, 1, None)
    softmax = np.exp(inv_scores) / np.sum(np.exp(inv_scores))

    out = pd.DataFrame({
        "driver": drivers,
        "predicted_pos": pred_pos,
        "probability_top3": softmax,
    })
    out.sort_values("predicted_pos", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Predict next race")
    parser.add_argument("--race", help="Race identifier e.g. 2025-Canada", default=None)
    args = parser.parse_args(argv)

    df = predict_next(args.race)
    print(df.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
