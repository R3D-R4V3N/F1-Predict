import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pytest
import requests

from f1_predictor.data_loader import DataLoader
from f1_predictor.feature_engineering import generate_feature_matrix
from f1_predictor import dataset_builder


class DummyResponse:
    def __init__(self, data: dict) -> None:
        self._text = json.dumps(data)

    @property
    def text(self) -> str:  # pragma: no cover - simple accessor
        return self._text

    def json(self) -> dict:
        return json.loads(self._text)

    def raise_for_status(self) -> None:  # pragma: no cover - no error
        pass


class DummyCSVResponse:
    def __init__(self, text: str) -> None:
        self.content = text.encode()

    def raise_for_status(self) -> None:  # pragma: no cover - no error
        pass


def test_cache_hit(tmp_path, monkeypatch) -> None:
    data = {"MRData": {"RaceTable": {"Races": [{"round": 1}]}}}

    def fake_get(self, url: str):
        return DummyResponse(data)

    monkeypatch.setattr(DataLoader, "_rate_limited_get", fake_get)
    loader = DataLoader(cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw")

    loader.fetch_season(2021)  # populate cache
    start = time.perf_counter()
    loader.fetch_season(2021)  # should hit cache
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms <= 10


def test_rollings() -> None:
    df = pd.DataFrame({"driver_id": [1] * 10, "score": np.arange(10)})
    df["driver_recent_avg"] = df.groupby("driver_id")["score"].apply(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    for i in range(len(df)):
        start = max(0, i - 5)
        expected = df.loc[start : i - 1, "score"].mean() if i > 0 else np.nan
        if np.isnan(expected):
            assert np.isnan(df.loc[i, "driver_recent_avg"])
        else:
            assert df.loc[i, "driver_recent_avg"] == pytest.approx(expected)


def test_tsplit_order() -> None:
    tscv = TimeSeriesSplit(n_splits=3)
    data = np.arange(10)
    for train_idx, val_idx in tscv.split(data):
        assert train_idx.max() < val_idx.min()
        assert list(train_idx) == sorted(train_idx)
        assert list(val_idx) == sorted(val_idx)


def test_racefans_fallback(tmp_path, monkeypatch) -> None:
    csv_data = "year,round,rating\n2019,1,7.5\n"

    calls = []

    def fake_get(self, url: str):
        calls.append(url)
        if "main" in url:
            raise requests.HTTPError("404")
        return DummyCSVResponse(csv_data)

    monkeypatch.setattr(DataLoader, "_rate_limited_get", fake_get)

    loader = DataLoader(cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw")
    rating = loader.fetch_racefans_rating(2019, 1)

    main_url = (
        "https://raw.githubusercontent.com/theoehrly/fastf1/main/docs/examples/"
        "racefans_driver_ratings.csv"
    )
    master_url = (
        "https://raw.githubusercontent.com/theoehrly/fastf1/master/docs/examples/"
        "racefans_driver_ratings.csv"
    )

    assert rating == 7.5
    assert calls == [main_url, master_url]


def test_extract_weather() -> None:
    loader = DataLoader(cache_dir="none", raw_dir="none")

    class DummySession:
        weather_data = pd.DataFrame(
            {"AirTemp": [22.5, 23.5, pd.NA], "Humidity": [55.0, pd.NA, 56.0]}
        )

    weather = loader.extract_weather(DummySession())

    assert weather["air_temp"] == pytest.approx(23.0)
    assert weather["humidity"] == pytest.approx(55.5)


def test_feature_matrix_with_weather() -> None:
    df = pd.DataFrame(
        {
            "year": [2021, 2021],
            "round": [1, 2],
            "driver_id": [44, 44],
            "Position": [1, 2],
            "air_temp": [20.0, 21.0],
            "humidity": [60.0, 58.0],
            "Points": [25, 18],
        }
    )

    X, _ = generate_feature_matrix(df)
    assert any("air_temp" in c for c in X.columns)
    assert any("humidity" in c for c in X.columns)


def test_build_dataset_creates_processed_dir(tmp_path, monkeypatch) -> None:
    class DummySession:
        results = pd.DataFrame({"DriverNumber": [1], "Position": [1]})

        def get_circuit_info(self):
            return {"Length": "5.0"}

    class DummyLoader:
        def __init__(self, *_, **__):
            pass

        def fetch_season(self, year: int) -> pd.DataFrame:
            return pd.DataFrame({"round": [1], "season": [year]})

        def parse_circuit_info(self, races: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"year": races["season"], "round": races["round"]})

        def fetch_session(self, year: int, rnd: int, session: str) -> DummySession:
            return DummySession()

        def fetch_pit_summary(self, year: int, rnd: int) -> pd.DataFrame:
            return pd.DataFrame({"driverId": [1]})

        def fetch_racefans_rating(self, year: int, rnd: int) -> float:
            return 7.0

        def extract_weather(self, session: DummySession) -> dict[str, float]:
            return {"air_temp": 20.0, "humidity": 55.0}

    processed_path = tmp_path / "data" / "processed" / "f1.parquet"
    monkeypatch.setattr(dataset_builder, "PROCESSED_PATH", processed_path)
    monkeypatch.setattr(dataset_builder, "DataLoader", DummyLoader)

    df = dataset_builder.build_dataset([2021])

    assert processed_path.exists()
    assert not df.empty
