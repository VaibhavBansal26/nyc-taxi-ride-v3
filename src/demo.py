"""Offline demo engine.

Replays a bundled month of real hourly ride counts (January 2024) onto the
current clock and runs the bundled LightGBM model locally.  This keeps the
Streamlit app fully functional without Hopsworks credentials and gives a
deterministic fixture for tests.
"""

from __future__ import annotations

import functools

import pandas as pd

import src.config as config
from src.data_utils import latest_window_features, transform_ts_data_info_features
from src.inference import get_model_predictions, load_local_model

SAMPLE_TILES = 2  # replay the month twice so monitoring can look back further than 72 h


@functools.lru_cache(maxsize=1)
def load_sample_ts() -> pd.DataFrame:
    """Bundled hourly rides with naive New-York local ``pickup_hour``."""
    if not config.SAMPLE_TS_FILE.exists():
        raise FileNotFoundError(f"Sample data missing: {config.SAMPLE_TS_FILE}")
    df = pd.read_parquet(config.SAMPLE_TS_FILE)
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    if df["pickup_hour"].dt.tz is not None:
        df["pickup_hour"] = df["pickup_hour"].dt.tz_convert(config.LOCAL_TIMEZONE).dt.tz_localize(None)
    return df[["pickup_location_id", "pickup_hour", "rides"]]


def replay_sample(target_hour: pd.Timestamp, tiles: int = SAMPLE_TILES) -> pd.DataFrame:
    """Shift the sample so its last hour is the hour before ``target_hour`` (UTC output)."""
    sample = load_sample_ts()
    span = sample["pickup_hour"].max() - sample["pickup_hour"].min() + pd.Timedelta(hours=1)
    parts = [sample.assign(pickup_hour=sample["pickup_hour"] - span * i) for i in range(tiles)]
    sample = pd.concat(parts, ignore_index=True)

    target_hour = pd.Timestamp(target_hour)
    target_hour = target_hour.tz_localize("UTC") if target_hour.tzinfo is None else target_hour.tz_convert("UTC")
    last_local = (target_hour - pd.Timedelta(hours=1)).tz_convert(config.LOCAL_TIMEZONE).tz_localize(None).floor("h")
    offset = last_local - sample["pickup_hour"].max()

    shifted = sample["pickup_hour"] + offset
    shifted = shifted.dt.tz_localize(config.LOCAL_TIMEZONE, nonexistent="shift_forward", ambiguous=False).dt.tz_convert(
        "UTC"
    )
    out = sample.assign(pickup_hour=shifted).dropna(subset=["pickup_hour"])
    out = out.drop_duplicates(["pickup_location_id", "pickup_hour"])
    return out.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _model():
    return load_local_model()


def demo_next_hour(target_hour: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(features, predictions)`` for the hour after ``target_hour``."""
    target_hour = pd.Timestamp(target_hour or pd.Timestamp.now(tz="UTC"))
    target_hour = (
        target_hour.tz_localize("UTC") if target_hour.tzinfo is None else target_hour.tz_convert("UTC")
    ).ceil("h")
    ts = replay_sample(target_hour)
    features = latest_window_features(ts, target_hour)
    preds = get_model_predictions(_model(), features)
    return features, preds


def demo_recent(hours: int, target_hour: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(actuals, predictions)`` for the ``hours`` hours before ``target_hour``.

    Predictions are produced the same way the hourly pipeline would have, one
    window per hour, so the monitoring page can compare them with actuals.
    """
    target_hour = pd.Timestamp(target_hour or pd.Timestamp.now(tz="UTC"))
    target_hour = (
        target_hour.tz_localize("UTC") if target_hour.tzinfo is None else target_hour.tz_convert("UTC")
    ).ceil("h")
    ts = replay_sample(target_hour)
    first_target = target_hour - pd.Timedelta(hours=hours)
    window_start = first_target - pd.Timedelta(hours=config.WINDOW_SIZE)
    subset = ts[ts["pickup_hour"] >= window_start]
    features = transform_ts_data_info_features(subset, step_size=1)
    features = features[pd.to_datetime(features["pickup_hour"]).dt.tz_convert("UTC") >= first_target]
    preds = (
        get_model_predictions(_model(), features)
        if not features.empty
        else pd.DataFrame(columns=["pickup_location_id", "pickup_hour", "predicted_demand"])
    )
    actuals = ts[ts["pickup_hour"] >= first_target][["pickup_location_id", "pickup_hour", "rides"]].reset_index(
        drop=True
    )
    return actuals, preds
