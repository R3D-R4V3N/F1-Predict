import pandas as pd

from f1_predictor import dataset_builder
from f1_predictor.data_loader import DataLoader


def test_build_dataset_parquet(tmp_path, monkeypatch):
    # Redirect dataset output
    out_path = tmp_path / "dataset.parquet"
    monkeypatch.setattr(dataset_builder, "PROCESSED_PATH", out_path)

    # Mock data loader methods
    def fake_fetch_season(self, year):
        return pd.DataFrame([
            {
                "season": year,
                "round": 1,
                "Circuit": {
                    "circuitId": "test",
                    "circuitName": "Test Circuit",
                    "Location": {
                        "locality": "Somewhere",
                        "country": "Neverland",
                        "lat": "10",
                        "long": "20",
                    },
                },
            }
        ])

    class DummySession:
        def __init__(self):
            self.results = pd.DataFrame({
                "DriverNumber": [44],
                "Position": [1],
                "Points": [25],
            })

        def get_circuit_info(self):
            return {"Length": 5.0}

    def fake_fetch_session(self, year, rnd, session="R"):
        return DummySession()

    monkeypatch.setattr(DataLoader, "fetch_season", fake_fetch_season)
    monkeypatch.setattr(DataLoader, "fetch_session", fake_fetch_session)
    monkeypatch.setattr(DataLoader, "fetch_pit_summary", lambda self, y, r: pd.DataFrame())
    monkeypatch.setattr(DataLoader, "fetch_racefans_rating", lambda self, y, r: 8.0)
    monkeypatch.setattr(DataLoader, "extract_weather", lambda self, s: {"air_temp": 20.0, "humidity": 60.0})

    df = dataset_builder.build_dataset([2022])

    assert out_path.exists()
    loaded = pd.read_parquet(out_path)
    assert set(["driver_id", "circuit_id", "track_length", "air_temp", "humidity"]).issubset(loaded.columns)
    assert df.equals(loaded)
