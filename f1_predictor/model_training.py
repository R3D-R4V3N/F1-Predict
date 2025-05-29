"""Model training utilities for Formula 1 predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor, Booster

from .feature_engineering import generate_feature_matrix

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Booster, pd.DataFrame]:
    """Train an XGBoost model with time series cross-validation.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector.

    Returns
    -------
    tuple[Booster, pd.DataFrame]
        Trained booster and cross-validation metrics.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
        }
        model = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            tree_method="hist",
            random_state=42,
            **params,
        )
        tscv = TimeSeriesSplit(n_splits=5)
        scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            mae = np.abs(preds - y.iloc[val_idx]).mean()
            scores.append(float(mae))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_trial.params
    LOGGER.info("Best params: %s", best_params)
    model = XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        **best_params,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    metrics: list[dict[str, float]] = []
    for i, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        mae = float(np.abs(preds - y.iloc[val_idx]).mean())
        metrics.append({"split": i, "mae": mae})

    model.fit(X, y)
    booster = model.get_booster()

    out_dir = MODELS_DIR / pd.Timestamp.today().strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.xgb"
    booster.save_model(str(model_path))
    LOGGER.info("Saved model to %s", model_path)

    metrics_df = pd.DataFrame(metrics)
    return booster, metrics_df


def _cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Formula 1 model")
    parser.add_argument(
        "--data",
        default="data/processed/f1_dataset.parquet",
        help="Path to processed dataset",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    df = pd.read_parquet(args.data)
    X, y = generate_feature_matrix(df)

    train_model(X, y)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
