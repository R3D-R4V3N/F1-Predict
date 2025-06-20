"""Utilities for downloading and caching Formula 1 data."""

from __future__ import annotations

import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests
import fastf1

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

LOGGER = logging.getLogger(__name__)


def retry(backoff: int = 2) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry a function up to three times with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = 1
            for attempt in range(3):
                try:
                    return func(*args, **kwargs)
                except Exception:  # pragma: no cover - runtime errors
                    if attempt == 2:
                        raise
                    LOGGER.exception(
                        "Error in %s, retrying in %s seconds", func.__name__, delay
                    )
                    time.sleep(delay)
                    delay *= backoff
            return None

        return wrapper

    return decorator


class DataLoader:
    """Handle downloading of data from various sources with caching."""

    _last_request: float = 0.0

    def __init__(
        self, cache_dir: str | Path | None = None, raw_dir: str | Path | None = None
    ) -> None:
        self.cache_dir = Path(cache_dir or ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path(raw_dir or "data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # FastF1 expects the cache directory to already exist. Ensure the
        # subfolder is created before enabling the cache to avoid a
        # ``NotADirectoryError`` on first run.
        fastf1_cache = self.cache_dir / "fastf1"
        fastf1_cache.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(fastf1_cache))

    def _rate_limited_get(self, url: str) -> requests.Response:
        wait_time = 0.25 - (time.time() - DataLoader._last_request)
        if wait_time > 0:
            time.sleep(wait_time)
        resp = requests.get(url, timeout=30)
        DataLoader._last_request = time.time()
        resp.raise_for_status()
        return resp

    def _download_json(self, url: str, dest: Path) -> Any:
        if dest.exists():
            with dest.open("r") as fh:
                return json.load(fh)

        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = self._rate_limited_get(url)
        with dest.open("w") as fh:
            fh.write(resp.text)
        return resp.json()

    def _download_csv(self, url: str, dest: Path) -> Path:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            resp = self._rate_limited_get(url)
            dest.write_bytes(resp.content)
        return dest

    @retry()
    def fetch_season(self, year: int) -> pd.DataFrame:
        """Fetch season schedule and results from Ergast."""
        url = f"https://ergast.com/api/f1/{year}.json?limit=1000"
        dest = self.raw_dir / "ergast" / f"season_{year}.json"
        try:
            data = self._download_json(url, dest)
        except Exception:
            LOGGER.info("Falling back to built-in schedule for %s", year)
            data = None

        races = []
        if data:
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

        if not races and year in FALLBACK_SCHEDULE:
            races = [
                {
                    "season": year,
                    "round": r["round"],
                    "raceName": r["raceName"],
                }
                for r in FALLBACK_SCHEDULE[year]
            ]
        return pd.DataFrame(races)

    @retry()
    def fetch_session(
        self, year: int, round: int, session: str = "R"
    ) -> fastf1.core.Session:
        """Fetch a FastF1 session."""
        s = fastf1.get_session(year, round, session)
        s.load()
        return s

    @retry()
    def fetch_racefans_rating(self, year: int, round: int) -> Optional[float]:
        """Fetch racefans driver rating for a specific race."""
        urls = [
            "https://raw.githubusercontent.com/theoehrly/fastf1/main/docs/examples/racefans_driver_ratings.csv",
            "https://raw.githubusercontent.com/theoehrly/fastf1/master/docs/examples/racefans_driver_ratings.csv",
        ]
        dest = self.raw_dir / "racefans" / "driver_ratings.csv"

        for url in urls:
            try:
                self._download_csv(url, dest)
                break
            except requests.HTTPError:  # pragma: no cover - network failure
                LOGGER.warning("Failed to fetch ratings from %s", url)
                if dest.exists():
                    dest.unlink()
        else:
            LOGGER.error("Unable to download racefans driver ratings")
            return None

        if not dest.exists():
            return None

        df = pd.read_csv(dest)
        row = df[(df["year"] == year) & (df["round"] == round)]
        if row.empty:
            return None
        return float(row.iloc[0]["rating"])

    @retry()
    def fetch_pit_summary(self, year: int, round: int) -> pd.DataFrame:
        """Fetch pit stop summary from Ergast."""
        url = f"https://ergast.com/api/f1/{year}/{round}/pitstops.json?limit=1000"
        dest = self.raw_dir / "ergast" / f"pits_{year}_{round}.json"
        data = self._download_json(url, dest)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        stops = races[0].get("PitStops", []) if races else []
        return pd.DataFrame(stops)

    def parse_circuit_info(self, races: pd.DataFrame) -> pd.DataFrame:
        """Extract circuit information from an Ergast schedule dataframe.

        Parameters
        ----------
        races:
            DataFrame returned by :meth:`fetch_season`.

        Returns
        -------
        pd.DataFrame
            Circuit details indexed by year and round.
        """

        rows: list[dict[str, Any]] = []
        for _, race in races.iterrows():
            circuit = race.get("Circuit", {}) or {}
            location = circuit.get("Location", {}) or {}
            try:
                year = int(race.get("season") or race.get("Season") or race.get("year"))
            except Exception:  # pragma: no cover - malformed data
                year = None
            try:
                rnd = int(race.get("round") or race.get("Round"))
            except Exception:
                continue
            rows.append(
                {
                    "year": year,
                    "round": rnd,
                    "circuit_id": circuit.get("circuitId"),
                    "circuit_name": circuit.get("circuitName"),
                    "circuit_locality": location.get("locality"),
                    "circuit_country": location.get("country"),
                    "circuit_lat": float(location["lat"]) if "lat" in location else None,
                    "circuit_long": float(location["long"]) if "long" in location else None,
                }
            )

        return pd.DataFrame(rows)

    def extract_weather(self, session: fastf1.core.Session) -> dict[str, Optional[float]]:
        """Return air temperature and humidity for a loaded session."""
        try:
            weather = session.weather_data  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fastf1 internals may fail
            weather = None

        if weather is None or getattr(weather, "empty", True):
            return {"air_temp": None, "humidity": None}

        air_series = pd.to_numeric(weather.get("AirTemp"), errors="coerce")
        hum_series = pd.to_numeric(weather.get("Humidity"), errors="coerce")

        air_mean = air_series.mean() if air_series is not None else float("nan")
        hum_mean = hum_series.mean() if hum_series is not None else float("nan")

        return {
            "air_temp": float(air_mean) if not pd.isna(air_mean) else None,
            "humidity": float(hum_mean) if not pd.isna(hum_mean) else None,
        }
