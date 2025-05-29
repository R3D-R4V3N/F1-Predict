"""Build a consolidated Formula 1 dataset using :class:`DataLoader`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import DataLoader

LOGGER = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/f1_dataset.parquet")


def build_dataset(seasons: Iterable[int]) -> pd.DataFrame:
    """Download and merge data for the given seasons.

    Parameters
    ----------
    seasons:
        Years to include in the dataset.

    Returns
    -------
    pd.DataFrame
        Merged dataset containing race results and basic metrics.
    """
    loader = DataLoader()
    frames: list[pd.DataFrame] = []

    for year in seasons:
        LOGGER.info("Fetching season %s", year)
        races = loader.fetch_season(year)
        if races.empty:
            continue
        for _, race in races.iterrows():
            try:
                rnd = int(race.get("round") or race.get("Round"))
            except Exception:
                continue
            LOGGER.info("Processing %s round %s", year, rnd)
            session = loader.fetch_session(year, rnd, "R")
            results = getattr(session, "results", pd.DataFrame()).copy()
            if results.empty:
                continue
            results["year"] = year
            results["round"] = rnd
            rating = loader.fetch_racefans_rating(year, rnd)
            results["racefans_rating"] = rating
            frames.append(results)

    if not frames:
        raise RuntimeError("No data collected")

    df = pd.concat(frames, ignore_index=True)
    if "DriverNumber" in df.columns:
        df.rename(columns={"DriverNumber": "driver_id"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "driver_id"}, inplace=True)

    df = df.drop_duplicates(subset=["year", "round", "driver_id"])
    df.to_parquet(PROCESSED_PATH, index=False)
    LOGGER.info("Saved dataset to %s", PROCESSED_PATH)
    return df


def _cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Formula 1 dataset")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="*",
        help="Seasons to include e.g. 2018 2019 2020",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    seasons = args.seasons
    if not seasons:
        seasons = list(range(2018, pd.Timestamp.today().year + 1))

    build_dataset(seasons)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
