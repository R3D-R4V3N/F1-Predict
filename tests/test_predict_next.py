import pandas as pd

from f1_predictor.predict_next_race import _parse_race_id
from f1_predictor.data_loader import DataLoader


def test_parse_slug_with_accents(monkeypatch):
    def fake_fetch_season(self, year):
        return pd.DataFrame([
            {"season": year, "round": 3, "raceName": "Gran Premio de España"}
        ])

    monkeypatch.setattr(DataLoader, "fetch_season", fake_fetch_season)

    assert _parse_race_id("2025-Spain") == (2025, 3)
