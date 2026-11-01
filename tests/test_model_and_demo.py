import numpy as np
import pandas as pd
import pytest

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target
from src.pipeline_utils import average_rides_last_4_weeks, get_pipeline

needs_assets = pytest.mark.skipif(
    not (config.LOCAL_MODEL_PATH.exists() and config.SAMPLE_TS_FILE.exists()), reason="bundled model/sample missing"
)


def test_average_rides_feature():
    X = pd.DataFrame({f"rides_t-{lag}": [lag] for lag in (168, 336, 504, 672)})
    assert average_rides_last_4_weeks(X)["average_rides_last_4_weeks"].item() == pytest.approx(420)


def test_pipeline_trains_and_predicts_on_synthetic_series():
    hours = config.WINDOW_SIZE + 24 * 10
    idx = pd.date_range("2024-03-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    rides = (20 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.normal(0, 1, hours)).round().clip(0)
    ts = pd.DataFrame({"pickup_location_id": 42, "pickup_hour": idx, "rides": rides})
    X, y = transform_ts_data_info_features_and_target(ts, step_size=4)
    model = get_pipeline(n_estimators=30, verbose=-1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.abs(preds - y).mean() < 5


@needs_assets
def test_demo_next_hour_predictions():
    from src.demo import demo_next_hour

    target = pd.Timestamp("2026-05-10 15:30", tz="UTC")
    features, preds = demo_next_hour(target)
    assert len(features) == len(preds) == 256
    assert (preds["pickup_hour"] == pd.Timestamp("2026-05-10 16:00", tz="UTC")).all()
    assert (preds["predicted_demand"] >= 0).all()
    assert preds["predicted_demand"].sum() > 0
    assert list(features.columns[-2:]) == ["pickup_location_id", "pickup_hour"]


@needs_assets
def test_demo_recent_aligns_actuals_and_predictions():
    from src.demo import demo_recent

    actuals, preds = demo_recent(24, pd.Timestamp("2026-05-10 15:30", tz="UTC"))
    merged = actuals.merge(preds, on=["pickup_location_id", "pickup_hour"])
    assert merged["pickup_hour"].nunique() == 24
    assert (merged["predicted_demand"] - merged["rides"]).abs().mean() < 15


@needs_assets
def test_resolve_source_demo():
    from src.data_source import resolve_source

    source, note = resolve_source("demo")
    assert source.mode == "demo" and note is None
