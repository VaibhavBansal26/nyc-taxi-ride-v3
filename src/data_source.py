"""Uniform data access for the app: live Hopsworks or offline demo.

All sources return the same frames:

* predictions: ``pickup_location_id, pickup_hour (UTC), predicted_demand``
* actuals:     ``pickup_location_id, pickup_hour (UTC), rides``
* features:    model layout (``rides_t-672 .. rides_t-1, pickup_location_id, pickup_hour``)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import src.config as config


@dataclass
class SourceStatus:
    mode: str  # "live" | "demo"
    detail: str


class HopsworksSource:
    mode = "live"

    def __init__(self):
        from src import inference

        self._inf = inference
        self.project_name = self._inf.get_hopsworks_project().name  # fails fast on bad credentials

    def status(self) -> SourceStatus:
        return SourceStatus("live", f"Hopsworks project '{self.project_name}'")

    def next_hour_predictions(self, target_hour: pd.Timestamp | None = None) -> pd.DataFrame:
        return self._inf.fetch_next_hour_predictions(target_hour)

    def recent_predictions(self, hours: int) -> pd.DataFrame:
        return self._inf.fetch_predictions(hours)

    def recent_actuals(self, hours: int) -> pd.DataFrame:
        return self._inf.fetch_hourly_rides(hours)

    def latest_features(self, target_hour: pd.Timestamp | None = None) -> pd.DataFrame:
        return self._inf.load_batch_of_features_from_store(target_hour)


class DemoSource:
    mode = "demo"

    def __init__(self):
        from src import demo

        self._demo = demo
        if not config.SAMPLE_TS_FILE.exists() or not config.LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError("Demo mode needs data/sample/*.parquet and models/lgb_model.pkl")

    def status(self) -> SourceStatus:
        return SourceStatus("demo", "offline replay of January 2024 rides through the bundled model")

    def next_hour_predictions(self, target_hour: pd.Timestamp | None = None) -> pd.DataFrame:
        _, preds = self._demo.demo_next_hour(target_hour)
        return preds

    def recent_predictions(self, hours: int) -> pd.DataFrame:
        _, preds = self._demo.demo_recent(hours)
        return preds

    def recent_actuals(self, hours: int) -> pd.DataFrame:
        actuals, _ = self._demo.demo_recent(hours)
        return actuals

    def latest_features(self, target_hour: pd.Timestamp | None = None) -> pd.DataFrame:
        features, _ = self._demo.demo_next_hour(target_hour)
        return features


def resolve_source(mode: str | None = None) -> tuple[HopsworksSource | DemoSource, str | None]:
    """Pick a source according to ``APP_MODE``; returns ``(source, warning_or_None)``."""
    mode = (mode or config.APP_MODE or "auto").lower()
    if mode == "demo":
        return DemoSource(), None
    if mode == "live":
        return HopsworksSource(), None
    if not config.hopsworks_configured():
        return DemoSource(), "HOPSWORKS_API_KEY is not set - running in demo mode."
    try:
        return HopsworksSource(), None
    except Exception as exc:
        return (
            DemoSource(),
            f"Could not connect to Hopsworks ({type(exc).__name__}: {str(exc)[:160]}). Running in demo mode.",
        )
