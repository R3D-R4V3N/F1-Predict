import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pytest

from f1_predictor.data_loader import DataLoader


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
    df["driver_recent_avg"] = (
        df.groupby("driver_id")["score"].apply(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )
    )
    for i in range(len(df)):
        start = max(0, i - 5)
        expected = df.loc[start:i - 1, "score"].mean() if i > 0 else np.nan
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

