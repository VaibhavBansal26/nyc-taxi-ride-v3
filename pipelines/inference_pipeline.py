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

# from datetime import timedelta
# import pandas as pd

# import src.config as config
# from src.inference import (
#     get_feature_store,
#     get_model_predictions,
#     load_model_from_registry,
#     ensure_feature_view,            # <-- ensure FV exists
# )
# from src.data_utils import transform_ts_data_info_features

# # Current time in UTC
# current_date = pd.Timestamp.now(tz="Etc/UTC")
# fs = get_feature_store()

# # read time-series data from the feature store
# fetch_data_to = current_date - timedelta(hours=1)
# fetch_data_from = current_date - timedelta(days=1 * 29)
# print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# # Make sure the Feature View exists (idempotent)
# fv = ensure_feature_view(fs)

# ts_data = fv.get_batch_data(
#     start_time=(fetch_data_from - timedelta(days=1)),
#     end_time=(fetch_data_to + timedelta(days=1)),
# )

# # Normalize to UTC BEFORE filtering
# ts_data = ts_data.copy()
# if "pickup_hour" not in ts_data.columns:
#     raise ValueError("inference_pipeline: 'pickup_hour' column missing in Feature View data.")
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
from datetime import timedelta
import pandas as pd

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
    ensure_feature_view,  # ensure FV exists (create if missing), returns FV
)
from src.data_utils import transform_ts_data_info_features


def _read_timeseries_with_fv_or_fg(fs, fetch_data_from: pd.Timestamp, fetch_data_to: pd.Timestamp) -> pd.DataFrame:
    """
    Prefer Feature View (auto-create if missing). If FV fails or returns no rows,
    fall back to reading the Feature Group directly.
    Returns columns: ['pickup_location_id', 'pickup_hour', 'rides'] filtered to [from,to] (UTC).
    """
    # 1) Try Feature View
    try:
        fv = ensure_feature_view(fs)  # may create it if missing
        ts_data = fv.get_batch_data(
            start_time=(fetch_data_from - timedelta(days=1)),
            end_time=(fetch_data_to + timedelta(days=1)),
        )
        if "pickup_hour" not in ts_data.columns:
            raise ValueError("[FV] 'pickup_hour' column missing in Feature View data")

        ts_data = ts_data.copy()
        ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"], errors="coerce", utc=True)
        ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

        if not ts_data.empty:
            print(f"[inference_pipeline] FV returned {len(ts_data)} rows "
                  f"({ts_data['pickup_hour'].min()} .. {ts_data['pickup_hour'].max()})")
            return ts_data
        else:
            print("[inference_pipeline] FV returned 0 rows in requested window; will try FG fallback.")
    except Exception as e:
        print(f"[inference_pipeline] FV error (falling back to FG): {type(e).__name__}: {e}")

    # 2) Fall back to Feature Group
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    df = fg.select_all().read()
    if "pickup_hour" not in df.columns:
        raise ValueError("[FG] 'pickup_hour' column missing in Feature Group data")

    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce", utc=True)
    df = df[df.pickup_hour.between(fetch_data_from, fetch_data_to)]

    print(
        f"[inference_pipeline] FG fallback returned {len(df)} rows "
        f"({df['pickup_hour'].min() if not df.empty else None} .. "
        f"{df['pickup_hour'].max() if not df.empty else None})"
    )
    return df


def main():
    # Current time in UTC
    current_date = pd.Timestamp.now(tz="Etc/UTC")
    fs = get_feature_store()

    # read time-series data from the store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"[inference_pipeline] Fetching data from {fetch_data_from} to {fetch_data_to}")

    ts_data = _read_timeseries_with_fv_or_fg(fs, fetch_data_from, fetch_data_to)

    if ts_data.empty:
        raise ValueError(
            "inference_pipeline: no rows from feature store in requested window. "
            "Check that the feature pipeline ingested data and that dates/timezones align."
        )

    ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

    # Build features: window large enough for 4 weekly lags; step_size=1 to match training
    features = transform_ts_data_info_features(
        ts_data,
        feature_col="rides",
        window_size=24 * 28,  # 672
        step_size=1,
        tz="America/New_York",
        fill_missing=True,
        fill_value=0.0,
    )

    if features.empty:
        raise ValueError(
            "inference_pipeline: no sliding windows created. "
            "Likely insufficient contiguous hourly data per location for a 672-hour window."
        )

    model = load_model_from_registry()
    preds = get_model_predictions(model, features)
    preds["pickup_hour"] = current_date.ceil("h")
    print(f"[inference_pipeline] Generated {len(preds)} predictions for {preds['pickup_hour'].iloc[0]}")

    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=1,
        description="Predictions from LGBM Model",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    fg.insert(preds, write_options={"wait_for_job": False})
    print("[inference_pipeline] Prediction insert submitted.")


if __name__ == "__main__":
    main()
