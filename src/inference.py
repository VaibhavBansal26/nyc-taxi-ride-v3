"""Hopsworks-backed data access: feature store reads, model registry, predictions.

Every function here needs a valid ``HOPSWORKS_API_KEY``.  The Streamlit app
goes through :mod:`src.data_source`, which falls back to the offline demo
engine (:mod:`src.demo`) when Hopsworks is not configured or unreachable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

import src.config as config
from src.data_utils import _to_utc, latest_window_features

PREDICTION_COLUMNS = ["pickup_location_id", "pickup_hour", "predicted_demand"]


# --------------------------------------------------------------------------- connection
def get_hopsworks_project():
    """Login (cached by the SDK) and return the project."""
    import hopsworks

    if not config.HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not set. Add it to .env or your deployment secrets.")
    kwargs: dict = {"api_key_value": config.HOPSWORKS_API_KEY}
    if config.HOPSWORKS_PROJECT_NAME:
        kwargs["project"] = config.HOPSWORKS_PROJECT_NAME
    if config.HOPSWORKS_HOST:
        kwargs["host"] = config.HOPSWORKS_HOST
    return hopsworks.login(**kwargs)


def get_feature_store():
    return get_hopsworks_project().get_feature_store()


def get_rides_feature_group(fs=None, create: bool = False):
    fs = fs or get_feature_store()
    if create:
        return fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            description="Hourly yellow-taxi pickups per TLC zone",
            primary_key=["pickup_location_id", "pickup_hour"],
            event_time="pickup_hour",
            online_enabled=False,
        )
    return fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)


def get_predictions_feature_group(fs=None, create: bool = False):
    fs = fs or get_feature_store()
    if create:
        return fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_MODEL_PREDICTION,
            version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
            description="Next-hour taxi demand predictions from the LightGBM model",
            primary_key=["pickup_location_id", "pickup_hour"],
            event_time="pickup_hour",
        )
    return fs.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION, version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION
    )


def ensure_feature_view(fs=None):
    """Return the feature view, creating it from the feature group if missing."""
    fs = fs or get_feature_store()
    try:
        fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
    except Exception:  # older SDKs raise when the view is missing, newer ones return None
        fv = None
    if fv is not None:
        return fv
    fg = get_rides_feature_group(fs)
    fs.create_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
        description="Hourly rides per pickup_location_id",
        query=fg.select_all(),
        labels=[],
    )
    print(f"[FV] Created {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
    fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
    if fv is None:
        raise RuntimeError(f"Feature view {config.FEATURE_VIEW_NAME} could not be created")
    return fv


# --------------------------------------------------------------------------- reads
def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def read_hourly_rides(start: pd.Timestamp, end: pd.Timestamp, fs=None) -> pd.DataFrame:
    """Hourly rides in ``[start, end]`` via the feature view, falling back to the feature group."""
    fs = fs or get_feature_store()
    start, end = _utc(start), _utc(end)
    df = None
    try:
        fv = ensure_feature_view(fs)
        df = fv.get_batch_data(start_time=start - pd.Timedelta(days=1), end_time=end + pd.Timedelta(days=1))
    except Exception as exc:  # pragma: no cover - depends on remote state
        print(f"[inference] feature view read failed ({type(exc).__name__}: {exc}); falling back to feature group")
    if df is None or df.empty or "pickup_hour" not in df.columns:
        fg = get_rides_feature_group(fs)
        df = fg.filter((fg.pickup_hour >= start.to_pydatetime()) & (fg.pickup_hour <= end.to_pydatetime())).read()
    if df is None or df.empty:
        return pd.DataFrame(columns=["pickup_location_id", "pickup_hour", "rides"])
    df = df.copy()
    df["pickup_hour"] = _to_utc(df["pickup_hour"])
    df = df[df["pickup_hour"].between(start, end)]
    df["pickup_location_id"] = df["pickup_location_id"].astype("int32")
    return (
        df[["pickup_location_id", "pickup_hour", "rides"]]
        .sort_values(["pickup_location_id", "pickup_hour"])
        .reset_index(drop=True)
    )


def fetch_hourly_rides(hours: int) -> pd.DataFrame:
    """Observed rides for the past ``hours`` hours."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return read_hourly_rides(now - pd.Timedelta(hours=hours), now)


def fetch_days_data(days: int) -> pd.DataFrame:
    """Observed rides for the past ``days`` days (used for retraining)."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return read_hourly_rides(now - pd.Timedelta(days=days), now)


def load_batch_of_features_from_store(current_date: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """One feature row per location for the hour after ``current_date``."""
    target_hour = _utc(current_date or pd.Timestamp.now(tz="UTC")).ceil("h")
    fetch_from = target_hour - pd.Timedelta(days=config.INFERENCE_LOOKBACK_DAYS)
    fetch_to = target_hour - pd.Timedelta(hours=1)
    print(f"[inference] reading rides {fetch_from} .. {fetch_to} for target {target_hour}")
    ts_data = read_hourly_rides(fetch_from, fetch_to)
    if ts_data.empty:
        raise ValueError("No rows in the feature store for the requested window. Has the feature pipeline run?")
    features = latest_window_features(ts_data, target_hour)
    if features.empty:
        raise ValueError("Sliding-window transform produced zero rows; check data continuity.")
    return features


def load_backfill_features(hours: int, current_date: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """One feature row per location **per hour** for the last ``hours`` target hours (plus the next one)."""
    from src.data_utils import transform_ts_data_info_features

    target_hour = _utc(current_date or pd.Timestamp.now(tz="UTC")).ceil("h")
    first_target = target_hour - pd.Timedelta(hours=hours)
    fetch_from = first_target - pd.Timedelta(hours=config.WINDOW_SIZE)
    ts_data = read_hourly_rides(fetch_from, target_hour - pd.Timedelta(hours=1))
    if ts_data.empty:
        raise ValueError("No rows in the feature store for the backfill window.")
    features = transform_ts_data_info_features(ts_data, step_size=1)
    keep = pd.to_datetime(features["pickup_hour"], utc=True) >= first_target
    return features[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- model
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Predict next-hour demand for each feature row (rounded, non-negative)."""
    if features is None or features.empty:
        raise ValueError("get_model_predictions: empty features DataFrame")
    if "pickup_hour" not in features.columns:
        raise ValueError("get_model_predictions: 'pickup_hour' required")
    preds = pd.Series(model.predict(features)).clip(lower=0).round(0)
    return pd.DataFrame(
        {
            "pickup_location_id": features["pickup_location_id"].astype("int32").reset_index(drop=True),
            "pickup_hour": _to_utc(features["pickup_hour"]).reset_index(drop=True),
            "predicted_demand": preds.astype(float),
        }
    )


def load_local_model(path: Path | None = None):
    """Load the bundled model (``models/lgb_model.pkl``)."""
    from src import pipeline_utils  # noqa: F401  (needed for unpickling)

    path = Path(path or config.LOCAL_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Local model not found at {path}")
    return joblib.load(path)


def _latest_registry_model(version: int | None = None):
    mr = get_hopsworks_project().get_model_registry()
    if version is not None:
        return mr.get_model(name=config.MODEL_NAME, version=version)
    models = mr.get_models(name=config.MODEL_NAME)
    if not models:
        raise LookupError(f"No model named {config.MODEL_NAME!r} in the registry")
    return max(models, key=lambda m: m.version)


def load_model_from_registry(version: int | None = None):
    """Download the newest registry model and unpickle it."""
    from src import pipeline_utils  # noqa: F401  (needed for unpickling)

    model = _latest_registry_model(version)
    model_dir = Path(model.download())
    candidates = list(model_dir.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl artefact in downloaded model dir {model_dir}")
    preferred = [c for c in candidates if c.name == "lgb_model.pkl"]
    return joblib.load((preferred or candidates)[0])


def load_metrics_from_registry(version: int | None = None) -> dict:
    try:
        return dict(_latest_registry_model(version).training_metrics or {})
    except LookupError:
        return {}


# --------------------------------------------------------------------------- predictions FG
def _normalise_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    out = df.copy()
    out["pickup_hour"] = _to_utc(out["pickup_hour"])
    out["pickup_location_id"] = out["pickup_location_id"].astype("int32")
    out["predicted_demand"] = pd.to_numeric(out["predicted_demand"], errors="coerce").clip(lower=0)
    return out[PREDICTION_COLUMNS].sort_values(["pickup_hour", "pickup_location_id"]).reset_index(drop=True)


def fetch_predictions(hours: int) -> pd.DataFrame:
    """Predictions whose target hour is within the past ``hours`` hours (plus the next hour)."""
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).floor("h")
    fg = get_predictions_feature_group()
    df = fg.filter(fg.pickup_hour >= since.to_pydatetime()).read()
    return _normalise_predictions(df)


def fetch_next_hour_predictions(target_hour: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predictions for the upcoming hour (``now`` rounded up)."""
    target_hour = _utc(target_hour or pd.Timestamp.now(tz="UTC")).ceil("h")
    df = fetch_predictions(hours=2)
    return df[df["pickup_hour"] == target_hour].reset_index(drop=True)


def write_predictions(preds: pd.DataFrame, wait: bool = False) -> None:
    fg = get_predictions_feature_group(create=True)
    fg.insert(_normalise_predictions(preds), write_options={"wait_for_job": wait})
