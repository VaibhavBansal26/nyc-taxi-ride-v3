import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    _month_year_iter,
    fill_missing_hours,
    filter_nyc_taxi_data,
    latest_window_features,
    transform_raw_data_into_ts_data,
    transform_ts_data_info_features,
    transform_ts_data_info_features_and_target,
)


def _ts(location: int, start: str, hours: int, rides=None) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    rides = np.arange(hours) if rides is None else rides
    return pd.DataFrame({"pickup_location_id": location, "pickup_hour": idx, "rides": rides})


def test_month_year_iter_spans_year_boundary():
    assert _month_year_iter(pd.Timestamp("2025-11-15"), pd.Timestamp("2026-01-02")) == [
        (2025, 11),
        (2025, 12),
        (2026, 1),
    ]


def test_filter_nyc_taxi_data_keeps_plausible_trips():
    base = pd.Timestamp("2024-01-10 08:00")
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": [base, base, base, base + pd.Timedelta(days=40), base],
            "tpep_dropoff_datetime": [base + pd.Timedelta(minutes=10)] * 2
            + [
                base - pd.Timedelta(minutes=1),
                base + pd.Timedelta(days=40, minutes=5),
                base + pd.Timedelta(minutes=10),
            ],
            "total_amount": [12.0, 15.0, 10.0, 10.0, 10.0],
            "PULocationID": [43, 264, 43, 43, 100],
        }
    )
    out = filter_nyc_taxi_data(raw, 2024, 1)
    assert list(out.columns) == ["pickup_datetime", "pickup_location_id"]
    assert out["pickup_location_id"].tolist() == [
        43,
        100,
    ]  # 264 excluded, negative duration dropped, next month dropped


def test_transform_raw_data_into_ts_data_counts_per_hour():
    raw = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                ["2024-01-01 10:05", "2024-01-01 10:55", "2024-01-01 11:00", "2024-01-01 10:30"]
            ).tz_localize("UTC"),
            "pickup_location_id": [1, 1, 1, 2],
        }
    )
    ts = transform_raw_data_into_ts_data(raw)
    assert ts["pickup_hour"].dt.tz is not None
    assert ts.set_index(["pickup_location_id", "pickup_hour"])["rides"].to_dict() == {
        (1, pd.Timestamp("2024-01-01 10:00", tz="UTC")): 2,
        (1, pd.Timestamp("2024-01-01 11:00", tz="UTC")): 1,
        (2, pd.Timestamp("2024-01-01 10:00", tz="UTC")): 1,
    }


def test_fill_missing_hours_creates_dense_grid():
    ts = _ts(7, "2024-01-01 00:00", 3).drop(index=1)
    dense = fill_missing_hours(ts)
    assert len(dense) == 3
    assert dense["rides"].tolist() == [0, 0, 2]


def test_features_and_target_small_window():
    ts = _ts(5, "2024-06-01 00:00", 6)  # rides 0..5
    X, y = transform_ts_data_info_features_and_target(ts, window_size=3, step_size=1, fill_missing=True)
    assert list(X.columns) == ["rides_t-3", "rides_t-2", "rides_t-1", "pickup_location_id", "pickup_hour"]
    assert X[["rides_t-3", "rides_t-2", "rides_t-1"]].values.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert y.tolist() == [3, 4, 5]
    assert str(X["pickup_hour"].dt.tz) == "America/New_York"
    assert X["pickup_hour"].iloc[0] == pd.Timestamp("2024-06-01 03:00", tz="UTC")


def test_features_step_size_and_short_series_skipped():
    ts = pd.concat([_ts(1, "2024-06-01 00:00", 10), _ts(2, "2024-06-01 00:00", 2)])
    X = transform_ts_data_info_features(ts, window_size=4, step_size=3)
    assert X["pickup_location_id"].unique().tolist() == [1]
    assert len(X) == 2  # windows starting at 0 and 3


def test_latest_window_features_pads_and_orders():
    target = pd.Timestamp("2024-06-02 00:00", tz="UTC")
    ts = pd.concat([_ts(1, "2024-06-01 20:00", 4, rides=[5, 6, 7, 8]), _ts(2, "2024-06-01 22:00", 1, rides=[9])])
    X = latest_window_features(ts, target, window_size=6)
    assert list(X.columns)[:6] == [f"rides_t-{k}" for k in range(6, 0, -1)]
    assert X.loc[X.pickup_location_id == 1, [f"rides_t-{k}" for k in range(6, 0, -1)]].values.tolist() == [
        [0, 0, 5, 6, 7, 8]
    ]
    assert X.loc[X.pickup_location_id == 2, "rides_t-2"].item() == 9
    assert (X["pickup_hour"] == target).all()


def test_empty_inputs():
    empty = pd.DataFrame(columns=["pickup_location_id", "pickup_hour", "rides"])
    assert transform_ts_data_info_features(empty).empty
    assert latest_window_features(empty, pd.Timestamp("2024-01-01", tz="UTC")).empty
    with pytest.raises(ValueError):
        transform_ts_data_info_features(_ts(1, "2024-01-01", 5), step_size=0)
