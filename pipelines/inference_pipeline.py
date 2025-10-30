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
# ---- put these near the top with your other imports ----
from datetime import datetime, timedelta, timezone
import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features

# ---------- helpers ----------
def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )

def get_feature_store() -> FeatureStore:
    return get_hopsworks_project().get_feature_store()

def _read_timeseries_from_store(
    feature_store: FeatureStore,
    fetch_data_from: pd.Timestamp,
    fetch_data_to: pd.Timestamp,
) -> pd.DataFrame:
    """
    Try Feature View; if it errors or returns no rows, fall back to Feature Group.
    Returns ['pickup_location_id','pickup_hour','rides'] in the requested window (UTC).
    """
    # 1) Try Feature View
    try:
        fv = feature_store.get_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
        )
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
        # swallow FV errors and try FG
        print(f"[store:FV] skipping due to error: {type(e).__name__}: {e}")

    # 2) Fallback to Feature Group
    fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    df = fg.select_all().read()
    if "pickup_hour" not in df.columns:
        raise ValueError("Feature Group read succeeded but 'pickup_hour' column is missing.")
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce", utc=True)
    ts = df[df.pickup_hour.between(fetch_data_from, fetch_data_to)]
    print(f"[store:FG] {len(ts)} rows. Range: {ts['pickup_hour'].min() if not ts.empty else None} .. {ts['pickup_hour'].max() if not ts.empty else None}")
    return ts

# ---------- replace your current function with this ----------
def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    fs = get_feature_store()

    # window the last 29 days up to one hour before current
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    ts_data = _read_timeseries_from_store(fs, fetch_data_from, fetch_data_to)
    if ts_data.empty:
        raise ValueError(
            "No rows returned from Feature View or Feature Group in the requested window."
        )

    ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

    # Build 672-hour sliding windows (28 days) with step 1 hour
    features = transform_ts_data_info_features(
        ts_data,
        feature_col="rides",
        window_size=24 * 28,   # 672
        step_size=1,
    )
    if features.empty:
        raise ValueError("Sliding-window transform produced zero rows; check data continuity.")

    # sanity
    for c in ["pickup_location_id", "pickup_hour"]:
        if c not in features.columns:
            raise ValueError(f"Missing required column after transform: {c}")

    return features
