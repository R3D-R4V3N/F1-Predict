"""Feature engineering utilities for Formula 1 data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path("models/feature_pipeline.joblib")


def _convert_time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert time strings to seconds."""
    try:
        return pd.to_timedelta(series).dt.total_seconds()
    except Exception:  # pragma: no cover - conversion errors
        return pd.to_numeric(series, errors="coerce")


def _ewm_feature(df: pd.DataFrame, col: str, alpha: float = 0.3) -> pd.Series:
    """Return exponential weighted mean for a column using shift to avoid leakage."""
    return (
        df.groupby("driver_id")[col]
        .apply(lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )


def generate_feature_matrix(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate feature matrix and target vector from raw race results."""
    if not {
        "year",
        "round",
        "driver_id",
        "Position",
    }.issubset(df_raw.columns):
        raise ValueError("Input dataframe missing required columns")

    df = df_raw.copy()

    if "GridPosition" in df.columns:
        df.rename(columns={"GridPosition": "grid_position"}, inplace=True)
    if "FastestLapTime" in df.columns:
        df["fastest_lap_time"] = _convert_time_to_seconds(df["FastestLapTime"])
    if "FastestLapSpeed" in df.columns:
        df.rename(columns={"FastestLapSpeed": "fastest_lap_speed"}, inplace=True)

    df.sort_values(["driver_id", "year", "round"], inplace=True)

    numeric_features: list[str] = []

    for col in [
        "Points",
        "Position",
        "fastest_lap_time",
        "fastest_lap_speed",
    ]:
        if col in df.columns:
            feat_name = f"{col.lower()}_ewm"
            df[feat_name] = _ewm_feature(df, col)
            numeric_features.append(feat_name)

    if "grid_position" in df.columns:
        numeric_features.append("grid_position")
    if "racefans_rating" in df.columns:
        numeric_features.append("racefans_rating")
    if "fastest_lap_time" in df.columns:
        numeric_features.append("fastest_lap_time")
    if "fastest_lap_speed" in df.columns:
        numeric_features.append("fastest_lap_speed")

    df["points_cumsum"] = (
        df.groupby("driver_id")["Points"].cumsum().shift(1)
    )
    numeric_features.append("points_cumsum")

    categorical_features = ["driver_id"]
    if "TeamName" in df.columns:
        categorical_features.append("TeamName")

    context_features = ["year", "round"]
    numeric_features.extend(context_features)

    feature_cols = numeric_features + categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X = preprocessor.fit_transform(df[feature_cols])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_PATH)

    feature_names = preprocessor.get_feature_names_out(feature_cols)
    X_df = pd.DataFrame(
        X.toarray() if hasattr(X, "toarray") else X,
        columns=feature_names,
        index=df.index,
    )

    y = df["Position"].astype(int)
    return X_df, y
