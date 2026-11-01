"""Data access and feature engineering helpers.

Conventions
-----------
* ``pickup_hour`` is always a tz-aware ``datetime64[ns, UTC]`` column once it
  leaves this module.  TLC trip timestamps are naive New-York local time and
  are localised before being converted to UTC.
* Sliding-window feature frames have the column order
  ``rides_t-<W> ... rides_t-1, pickup_location_id, pickup_hour`` which the
  trained model relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

from src.config import (
    LOCAL_TIMEZONE,
    RAW_DATA_DIR,
    TLC_TRIP_DATA_URL,
    WINDOW_SIZE,
    ZONE_LOOKUP_FILE,
)

TS_COLUMNS = ["pickup_location_id", "pickup_hour", "rides"]
EXCLUDED_LOCATION_IDS = (1, 264, 265)  # Newark airport + "unknown" zones


# --------------------------------------------------------------------------- zones
def load_zone_lookup() -> pd.DataFrame:
    """Taxi-zone lookup with columns ``pickup_location_id, zone, borough``."""
    df = pd.read_csv(ZONE_LOOKUP_FILE)
    df = df.rename(columns={"LocationID": "pickup_location_id", "Zone": "zone", "Borough": "borough"})
    df["pickup_location_id"] = df["pickup_location_id"].astype("int32")
    return df[["pickup_location_id", "zone", "borough"]]


# --------------------------------------------------------------------------- raw trips
def fetch_raw_trip_data(year: int, month: int, timeout: int = 120) -> Path:
    """Download one month of yellow-taxi trips to ``data/raw`` and return the path."""
    url = TLC_TRIP_DATA_URL.format(year=year, month=month)
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise FileNotFoundError(f"{url} is not available (HTTP {response.status_code})")
    path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def filter_nyc_taxi_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Keep plausible trips inside ``year-month``.

    Returns ``['pickup_datetime', 'pickup_location_id']`` with *naive* local timestamps.
    """
    if not 1 <= month <= 12:
        raise ValueError("Month must be 1..12")
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.DateOffset(months=1)

    duration = rides["tpep_dropoff_datetime"] - rides["tpep_pickup_datetime"]
    duration_ok = (duration > pd.Timedelta(0)) & (duration <= pd.Timedelta(hours=5))
    amount_ok = (rides["total_amount"] > 0) & (rides["total_amount"] <= rides["total_amount"].quantile(0.999))
    nyc_ok = ~rides["PULocationID"].isin(EXCLUDED_LOCATION_IDS)
    month_ok = (rides["tpep_pickup_datetime"] >= start) & (rides["tpep_pickup_datetime"] < end)

    out = rides.loc[duration_ok & amount_ok & nyc_ok & month_ok, ["tpep_pickup_datetime", "PULocationID"]]
    out = out.rename(columns={"tpep_pickup_datetime": "pickup_datetime", "PULocationID": "pickup_location_id"})
    if out.empty:
        raise ValueError(f"No valid rides for {year}-{month:02}")
    return out.reset_index(drop=True)


def _month_year_iter(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[int, int]]:
    start_ts, end_ts = pd.Timestamp(start_ts), pd.Timestamp(end_ts)
    start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)  # periods are tz-naive
    months = pd.period_range(start_ts.to_period("M"), end_ts.to_period("M"), freq="M")
    return [(p.year, p.month) for p in months]


def _localize_naive_ny(series: pd.Series) -> pd.Series:
    """Naive NY-local timestamps -> tz-aware UTC."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(LOCAL_TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    return ts.dt.tz_convert("UTC")


def fetch_batch_raw_data(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Month-aware fetch of trips in ``[start_ts, end_ts]`` (UTC).

    TLC publishes trip files with roughly a two-month delay.  When the requested
    month is not available yet, the same month of the previous year is used and
    shifted forward by one year so the project can run as a "live" simulation.
    """
    start_ts = pd.to_datetime(start_ts, utc=True)
    end_ts = pd.to_datetime(end_ts, utc=True)

    frames: list[pd.DataFrame] = []
    for year, month in _month_year_iter(start_ts, end_ts):
        target_fp = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
        used_fallback = False
        if not target_fp.exists():
            try:
                print(f"[fetch_batch_raw_data] downloading {year}-{month:02}")
                fetch_raw_trip_data(year, month)
            except Exception:
                prev_fp = RAW_DATA_DIR / f"rides_{year - 1}_{month:02}.parquet"
                if not prev_fp.exists():
                    print(f"[fetch_batch_raw_data] {year}-{month:02} unavailable; using {year - 1}-{month:02}")
                    fetch_raw_trip_data(year - 1, month)
                target_fp, used_fallback = prev_fp, True

        raw = pd.read_parquet(target_fp, engine="pyarrow")
        monthly = filter_nyc_taxi_data(raw, year - 1 if used_fallback else year, month)
        if used_fallback:
            monthly["pickup_datetime"] = monthly["pickup_datetime"] + pd.DateOffset(years=1)
        monthly["pickup_datetime"] = _localize_naive_ny(monthly["pickup_datetime"])
        frames.append(monthly.dropna(subset=["pickup_datetime"]))

    if not frames:
        return pd.DataFrame(columns=["pickup_datetime", "pickup_location_id"])

    combined = pd.concat(frames, ignore_index=True)
    mask = combined["pickup_datetime"].between(start_ts, end_ts)
    return combined.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- hourly aggregation
def transform_raw_data_into_ts_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips into hourly ride counts per pickup zone (UTC hours)."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=TS_COLUMNS)
    missing = {"pickup_datetime", "pickup_location_id"} - set(raw_df.columns)
    if missing:
        raise ValueError(f"transform_raw_data_into_ts_data needs columns {sorted(missing)}")

    ts = pd.to_datetime(raw_df["pickup_datetime"], errors="coerce")
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df = pd.DataFrame(
        {
            "pickup_location_id": raw_df["pickup_location_id"].reset_index(drop=True),
            "pickup_hour": ts.dt.floor("h").reset_index(drop=True),
        }
    )
    out = df.groupby(["pickup_location_id", "pickup_hour"]).size().reset_index(name="rides")
    out["pickup_location_id"] = pd.to_numeric(out["pickup_location_id"], errors="coerce").fillna(0).astype("int32")
    out["rides"] = out["rides"].astype("int32")
    return out.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)


def fill_missing_hours(
    ts_df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Return a dense hourly grid per location, missing hours filled with 0 rides."""
    if ts_df.empty:
        return ts_df.copy()
    df = ts_df.copy()
    df["pickup_hour"] = _to_utc(df["pickup_hour"])
    start = df["pickup_hour"].min() if start is None else pd.Timestamp(start).tz_convert("UTC")
    end = df["pickup_hour"].max() if end is None else pd.Timestamp(end).tz_convert("UTC")
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    locations = np.sort(df["pickup_location_id"].unique())
    grid = pd.MultiIndex.from_product([locations, hours], names=["pickup_location_id", "pickup_hour"])
    dense = df.set_index(["pickup_location_id", "pickup_hour"])["rides"].reindex(grid, fill_value=0).reset_index()
    dense["pickup_location_id"] = dense["pickup_location_id"].astype("int32")
    dense["rides"] = dense["rides"].astype("int32")
    return dense


def _to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")


# --------------------------------------------------------------------------- sliding windows
def _feature_columns(feature_col: str, window_size: int) -> list[str]:
    return [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]


def _windows_for_location(values: np.ndarray, times: np.ndarray, window_size: int, step_size: int):
    """Vectorised sliding windows -> (features, targets, target_times)."""
    if values.shape[0] < window_size + 1:
        return None
    windows = sliding_window_view(values, window_size + 1)[::step_size]
    return windows[:, :window_size], windows[:, window_size], times[window_size:][::step_size]


def _prepare_for_windows(df: pd.DataFrame, feature_col: str, tz: str, fill_missing: bool, fill_value: float | None):
    required = {"pickup_location_id", "pickup_hour", feature_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")
    df = df[["pickup_location_id", "pickup_hour", feature_col]].copy()
    df["pickup_hour"] = _to_utc(df["pickup_hour"]).dt.tz_convert(tz)
    df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
    df = df.dropna(subset=["pickup_hour", feature_col]).sort_values(["pickup_location_id", "pickup_hour"])

    for location_id, g in df.groupby("pickup_location_id", sort=False):
        g = g.drop_duplicates("pickup_hour").set_index("pickup_hour").sort_index()
        if fill_missing:
            full_index = pd.date_range(g.index.min(), g.index.max(), freq="h", tz=tz)
            g = g.reindex(full_index)
            g[feature_col] = (
                g[feature_col].fillna(fill_value) if fill_value is not None else g[feature_col].ffill().bfill()
            )
        yield location_id, g[feature_col].to_numpy(dtype=float), g.index.to_numpy()


def transform_ts_data_info_features_and_target(
    df: pd.DataFrame,
    feature_col: str = "rides",
    window_size: int = WINDOW_SIZE,
    step_size: int = 1,
    tz: str = LOCAL_TIMEZONE,
    fill_missing: bool = True,
    fill_value: float | None = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sliding-window features **and** the next-hour target for training.

    Returns ``(features, targets)`` where ``features`` has the model column
    order and ``pickup_hour`` is the (tz-aware) hour being predicted.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=_feature_columns(feature_col, window_size) + ["pickup_location_id", "pickup_hour"]
        ), pd.Series(dtype=float, name="target")
    if step_size is None or step_size <= 0:
        raise ValueError(f"step_size must be positive; got {step_size}")

    feature_columns = _feature_columns(feature_col, window_size)
    feature_parts, target_parts, meta_parts = [], [], []
    for location_id, values, times in _prepare_for_windows(df, feature_col, tz, fill_missing, fill_value):
        res = _windows_for_location(values, times, window_size, step_size)
        if res is None:
            continue
        X, y, t = res
        feature_parts.append(X)
        target_parts.append(y)
        meta_parts.append(pd.DataFrame({"pickup_location_id": location_id, "pickup_hour": t}))

    if not feature_parts:
        return pd.DataFrame(columns=feature_columns + ["pickup_location_id", "pickup_hour"]), pd.Series(
            dtype=float, name="target"
        )

    features = pd.DataFrame(np.vstack(feature_parts), columns=feature_columns)
    meta = pd.concat(meta_parts, ignore_index=True)
    features["pickup_location_id"] = meta["pickup_location_id"].astype("int32").values
    features["pickup_hour"] = pd.to_datetime(meta["pickup_hour"], utc=True).dt.tz_convert(tz)
    targets = pd.Series(np.concatenate(target_parts), name="target")
    return features, targets


def transform_ts_data_info_features(
    df: pd.DataFrame,
    feature_col: str = "rides",
    window_size: int = WINDOW_SIZE,
    step_size: int = 1,
    tz: str = LOCAL_TIMEZONE,
    fill_missing: bool = True,
    fill_value: float | None = 0.0,
) -> pd.DataFrame:
    """Sliding-window features only (every window, ``step_size`` apart)."""
    features, _ = transform_ts_data_info_features_and_target(
        df, feature_col, window_size, step_size, tz, fill_missing, fill_value
    )
    return features


def latest_window_features(
    ts_df: pd.DataFrame,
    target_hour: pd.Timestamp,
    feature_col: str = "rides",
    window_size: int = WINDOW_SIZE,
    tz: str = LOCAL_TIMEZONE,
) -> pd.DataFrame:
    """One feature row per location for predicting ``target_hour``.

    The window covers the ``window_size`` hours strictly before ``target_hour``.
    Hours with no observed rides count as 0; locations with no data at all are
    skipped.
    """
    if ts_df is None or ts_df.empty:
        return pd.DataFrame(columns=_feature_columns(feature_col, window_size) + ["pickup_location_id", "pickup_hour"])
    target_hour = pd.Timestamp(target_hour)
    target_hour = target_hour.tz_localize("UTC") if target_hour.tzinfo is None else target_hour.tz_convert("UTC")
    target_hour = target_hour.floor("h")
    window_start = target_hour - pd.Timedelta(hours=window_size)
    window_end = target_hour - pd.Timedelta(hours=1)

    dense = fill_missing_hours(ts_df, start=window_start, end=window_end)
    dense = dense[dense["pickup_hour"].between(window_start, window_end)]
    pivot = dense.pivot(index="pickup_location_id", columns="pickup_hour", values=feature_col).fillna(0)
    pivot = pivot.reindex(columns=pd.date_range(window_start, window_end, freq="h", tz="UTC"), fill_value=0)

    features = pd.DataFrame(pivot.to_numpy(dtype=float), columns=_feature_columns(feature_col, window_size))
    features["pickup_location_id"] = pivot.index.astype("int32").values
    features["pickup_hour"] = target_hour.tz_convert(tz)
    return features.reset_index(drop=True)
