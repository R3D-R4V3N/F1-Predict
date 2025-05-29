"""Build a consolidated Formula 1 dataset using :class:`DataLoader`."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import DataLoader

LOGGER = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/f1_dataset.parquet")


def build_dataset(seasons: Iterable[int], max_attempts: int = 5) -> pd.DataFrame:
"""Download and merge data for the given seasons.

    Parameters
    ----------
    seasons:
        Years to include in the dataset.

    max_attempts:
        Maximum number of retries when fetching data before skipping a race
        or season.

    Returns
    -------
    pd.DataFrame
        Merged dataset containing race results and basic metrics.
    """
    loader = DataLoader()
    frames: list[pd.DataFrame] = []
    processed_races: set[tuple[int, int]] = set()

    if PROCESSED_PATH.exists():
        existing = pd.read_parquet(PROCESSED_PATH)
        frames.append(existing)
        processed_races = set(
            zip(existing["year"].astype(int), existing["round"].astype(int))
        )
        LOGGER.info("Loaded existing dataset with %s rows", len(existing))

    for year in seasons:
        LOGGER.info("Fetching season %s", year)
        races = pd.DataFrame()
        for attempt in range(max_attempts):
            try:
                races = loader.fetch_season(year)
                break
            except Exception:
                LOGGER.exception(
                    "Failed to fetch season %s, retrying in 5 seconds", year
                )
                time.sleep(5)
        else:
            LOGGER.error(
                "Could not fetch season %s after %s attempts", year, max_attempts
            )
        if races.empty:
            continue
        circuit_df = loader.parse_circuit_info(races)
        circuit_map = {
            (int(row["year"]), int(row["round"])): row for _, row in circuit_df.iterrows()
        }
        for _, race in races.iterrows():
            try:
                rnd = int(race.get("round") or race.get("Round"))
            except Exception:
                continue
            if (year, rnd) in processed_races:
                LOGGER.info("Skipping %s round %s (already processed)", year, rnd)
                continue
            LOGGER.info("Processing %s round %s", year, rnd)
            skip_race = False
            session = None
            for attempt in range(max_attempts):
                try:
                    session = loader.fetch_session(year, rnd, "R")
                    break
                except ValueError:
                    LOGGER.exception(
                        "Failed to fetch session %s round %s due to invalid data",
                        year,
                        rnd,
                    )
                    time.sleep(1)
                    skip_race = True
                    break
                except Exception:
                    LOGGER.exception(
                        "Failed to fetch session %s round %s, retrying in 5 seconds",
                        year,
                        rnd,
                    )
                    time.sleep(5)
            else:
                LOGGER.error(
                    "Could not fetch session %s round %s after %s attempts",
                    year,
                    rnd,
                    max_attempts,
                )
                skip_race = True
            if skip_race or session is None:
                continue
            results = getattr(session, "results", pd.DataFrame()).copy()
            if results.empty:
                continue
            # merge pit stop counts
            try:
                pits = loader.fetch_pit_summary(year, rnd)
            except Exception:  # pragma: no cover - network/runtime errors
                LOGGER.exception(
                    "Failed to fetch pit summary for %s round %s", year, rnd
                )
                pits = pd.DataFrame()
            if not pits.empty and "driverId" in pits.columns:
                counts = (
                    pits.groupby("driverId").size().reset_index(name="pit_stops")
                )
                merge_col = None
                for col in ("driverId", "Driver", "driver"):
                    if col in results.columns:
                        merge_col = col
                        break
                if merge_col:
                    results = results.merge(
                        counts, how="left", left_on=merge_col, right_on="driverId"
                    )
                    if merge_col != "driverId":
                        results.drop(columns=["driverId"], inplace=True)
                else:
                    results["pit_stops"] = counts["pit_stops"].mean()
            elif "pit_stops" not in results.columns:
                results["pit_stops"] = pd.NA
            results["year"] = year
            results["round"] = rnd
            circ_info = circuit_map.get((year, rnd))
            if circ_info is not None:
                for col, val in circ_info.items():
                    results[col] = val

            track_len = None
            try:
                ci = getattr(session, "get_circuit_info", None)
                ci = ci() if callable(ci) else getattr(session, "circuit_info", None)
                if ci:
                    if isinstance(ci, dict):
                        track_len = ci.get("Length") or ci.get("length")
                    else:
                        track_len = getattr(ci, "Length", None) or getattr(ci, "length", None)
                    if isinstance(track_len, str):
                        track_len = track_len.replace("km", "").strip()
                    track_len = float(track_len)
            except Exception:  # pragma: no cover - unsupported structure
                track_len = None
            results["track_length"] = track_len
            rating = loader.fetch_racefans_rating(year, rnd)
            results["racefans_rating"] = rating
            weather = loader.extract_weather(session)
            results["air_temp"] = weather.get("air_temp")
            results["humidity"] = weather.get("humidity")
            frames.append(results)
            processed_races.add((year, rnd))

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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum retries for downloads",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    seasons = args.seasons
    if not seasons:
        seasons = list(range(2018, pd.Timestamp.today().year + 1))

    build_dataset(seasons, max_attempts=args.max_attempts)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
