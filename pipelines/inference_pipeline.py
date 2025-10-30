# from datetime import datetime, timedelta

# import pandas as pd

# import src.config as config
# from src.inference import (
#     get_feature_store,
#     get_model_predictions,
#     load_model_from_registry,
# )

# # Get the current datetime64[us, Etc/UTC]
# # for number in range(22, 24 * 29):
# # current_date = pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=number)
# current_date = pd.Timestamp.now(tz="Etc/UTC")
# feature_store = get_feature_store()

# # read time-series data from the feature store
# fetch_data_to = current_date - timedelta(hours=1)
# fetch_data_from = current_date - timedelta(days=1 * 29)
# print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
# feature_view = feature_store.get_feature_view(
#     name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
# )

# ts_data = feature_view.get_batch_data(
#     start_time=(fetch_data_from - timedelta(days=1)),
#     end_time=(fetch_data_to + timedelta(days=1)),
# )
# ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
# ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)
# ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

# from src.data_utils import transform_ts_data_info_features

# features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)

# model = load_model_from_registry()

# predictions = get_model_predictions(model, features)
# predictions["pickup_hour"] = current_date.ceil("h")
# print(predictions)

# feature_group = get_feature_store().get_or_create_feature_group(
#     name=config.FEATURE_GROUP_MODEL_PREDICTION,
#     version=1,
#     description="Predictions from LGBM Model",
#     primary_key=["pickup_location_id", "pickup_hour"],
#     event_time="pickup_hour",
# )

# feature_group.insert(predictions, write_options={"wait_for_job": False})
# from datetime import timedelta
# import pandas as pd

# import src.config as config
# from src.inference import (
#     get_feature_store,
#     get_model_predictions,
#     load_model_from_registry,
# )
# from src.data_utils import transform_ts_data_info_features

# # Current time in UTC
# current_date = pd.Timestamp.now(tz="Etc/UTC")
# fs = get_feature_store()

# # read time-series data from the feature store
# fetch_data_to = current_date - timedelta(hours=1)
# fetch_data_from = current_date - timedelta(days=1 * 29)
# print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)

# ts_data = fv.get_batch_data(
#     start_time=(fetch_data_from - timedelta(days=1)),
#     end_time=(fetch_data_to + timedelta(days=1)),
# )

# # Normalize to UTC BEFORE filtering
# ts_data = ts_data.copy()
# ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"], errors="coerce", utc=True)
# ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

# if ts_data.empty:
#     raise ValueError(
#         "inference_pipeline: no rows from feature store in requested window. "
#         "Check dates/timezones or widen the fetch window."
#     )

# ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

# # Build features: window large enough for 4 weekly lags; step_size=1 to match training
# features = transform_ts_data_info_features(
#     ts_data,
#     feature_col="rides",
#     window_size=24 * 28,  # 672
#     step_size=1,
# )

# if features.empty:
#     raise ValueError(
#         "inference_pipeline: no sliding windows created. "
#         "Likely insufficient contiguous hourly data per location for a 672-hour window."
#     )

# model = load_model_from_registry()
# preds = get_model_predictions(model, features)
# preds["pickup_hour"] = current_date.ceil("h")
# print(preds)

# fg = get_feature_store().get_or_create_feature_group(
#     name=config.FEATURE_GROUP_MODEL_PREDICTION,
#     version=1,
#     description="Predictions from LGBM Model",
#     primary_key=["pickup_location_id", "pickup_hour"],
#     event_time="pickup_hour",
# )
# fg.insert(preds, write_options={"wait_for_job": False})

# pipelines/inference_pipeline.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features
from src.pipeline_utils import (
    TemporalFeatureEngineer,            # needed for joblib unpickle
    average_rides_last_4_weeks,         # needed for joblib unpickle
    ensure_required_lag_features,
    REQUIRED_LAGS_FOR_AVG_4W,
)

# ---------- Hopsworks plumbing ----------
def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )

def get_feature_store() -> FeatureStore:
    return get_hopsworks_project().get_feature_store()

# ---------- FV auto-create (idempotent) ----------
def _ensure_feature_view(fs: FeatureStore):
    """
    Ensure Feature View exists pointing to your hourly Feature Group.
    - If FV already exists -> returns it.
    - If not, attempts to create it from FG.
    - If creation fails (permissions/roles), returns None (caller will fallback to FG).
    """
    try:
        fv = fs.get_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
        )
        return fv
    except Exception:
        pass  # will try to create

    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        query = fg.select_all()
        # create if missing
        fs.create_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
            description="Hourly rides per pickup_location_id (auto-created)",
            query=query,
            labels=[],
        )
        # fetch again
        fv = fs.get_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
        )
        print(f"[FV] Created: {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
        return fv
    except Exception as e:
        print(f"[FV] Could not create FV (will fallback to FG): {type(e).__name__}: {e}")
        return None

# ---------- Reader with FV>FG fallback ----------
def _read_timeseries_from_store(
    feature_store: FeatureStore,
    fetch_data_from: pd.Timestamp,
    fetch_data_to: pd.Timestamp,
) -> pd.DataFrame:
    """
    Prefer Feature View (auto-create if missing). If FV is unavailable or empty, fallback to Feature Group.
    Returns: ['pickup_location_id','pickup_hour','rides'] (UTC) within [from, to].
    """
    # Try FV (and create if missing)
    fv = None
    try:
        fv = _ensure_feature_view(feature_store)
        if fv is not None:
            tmp = fv.get_batch_data(
                start_time=(fetch_data_from - timedelta(days=1)),
                end_time=(fetch_data_to + timedelta(days=1)),
            )
            if "pickup_hour" in tmp.columns:
                tmp = tmp.copy()
                tmp["pickup_hour"] = pd.to_datetime(tmp["pickup_hour"], errors="coerce", utc=True)
                ts = tmp[tmp.pickup_hour.between(fetch_data_from, fetch_data_to)]
                if not ts.empty:
                    print(f"[store:FV] {len(ts)} rows. Range: {ts['pickup_hour'].min()} .. {ts['pickup_hour'].max()}")
                    return ts
    except Exception as e:
        print(f"[store:FV] error (will fallback to FG): {type(e).__name__}: {e}")

    # FG fallback
    fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    df = fg.select_all().read()
    if "pickup_hour" not in df.columns:
        raise ValueError("Feature Group read ok but 'pickup_hour' column missing.")
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce", utc=True)
    ts = df[df.pickup_hour.between(fetch_data_from, fetch_data_to)]
    print(f"[store:FG] {len(ts)} rows. Range: "
          f"{ts['pickup_hour'].min() if not ts.empty else None} .. "
          f"{ts['pickup_hour'].max() if not ts.empty else None}")
    return ts

# ---------- Public loader used by app/pipeline ----------
def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    fs = get_feature_store()
    # window: last 29 days up to one hour before now
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    ts_data = _read_timeseries_from_store(fs, fetch_data_from, fetch_data_to)
    if ts_data.empty:
        raise ValueError("No rows in requested window from FV or FG.")

    ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

    # 28d = 672 hours
    features = transform_ts_data_info_features(
        ts_data,
        feature_col="rides",
        window_size=24 * 28,
        step_size=1,
    )
    if features.empty:
        raise ValueError("Sliding-window transform produced zero rows; check data continuity.")

    # sanity columns needed by the pipeline
    for c in ["pickup_location_id", "pickup_hour"]:
        if c not in features.columns:
            raise ValueError(f"Missing required column after transform: {c}")

    return features

# ---------- (unchanged) model helpers ----------
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    if features is None or features.empty:
        raise ValueError("get_model_predictions: empty features DataFrame")

    # ensure weekly-average transformer inputs exist
    features = ensure_required_lag_features(
        features, feature_col="rides",
        required_lags=REQUIRED_LAGS_FOR_AVG_4W, fill_value=0.0
    )

    if "pickup_hour" not in features.columns:
        raise ValueError("get_model_predictions: 'pickup_hour' missing before model.predict")

    if not pd.api.types.is_datetime64_any_dtype(features["pickup_hour"]):
        features = features.copy()
        features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], errors="coerce", utc=True)

    preds = model.predict(features)
    out = pd.DataFrame({
        "pickup_location_id": features["pickup_location_id"].values,
        "predicted_demand": pd.Series(preds).round(0),
    })
    return out

def load_model_from_registry(version=None):
    import joblib
    from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks  # noqa: F401
    mr = get_hopsworks_project().get_model_registry()
    model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
    model_dir = model.download()
    return joblib.load(Path(model_dir) / "lgb_model.pkl")

def load_metrics_from_registry(version=None):
    mr = get_hopsworks_project().get_model_registry()
    model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
    return model.training_metrics
