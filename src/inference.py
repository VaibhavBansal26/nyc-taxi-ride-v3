# from datetime import datetime, timedelta, timezone

# import hopsworks
# import numpy as np
# import pandas as pd
# from hsfs.feature_store import FeatureStore

# import src.config as config
# from src.data_utils import transform_ts_data_info_features


# def get_hopsworks_project() -> hopsworks.project.Project:
#     return hopsworks.login(
#         project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
#     )


# def get_feature_store() -> FeatureStore:
#     project = get_hopsworks_project()
#     return project.get_feature_store()


# def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
#     # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
#     predictions = model.predict(features)

#     results = pd.DataFrame()
#     results["pickup_location_id"] = features["pickup_location_id"].values
#     results["predicted_demand"] = predictions.round(0)

#     return results


# def load_batch_of_features_from_store(
#     current_date: datetime,
# ) -> pd.DataFrame:
#     feature_store = get_feature_store()

#     # read time-series data from the feature store
#     fetch_data_to = current_date - timedelta(hours=1)
#     fetch_data_from = current_date - timedelta(days=29)
#     print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
#     feature_view = feature_store.get_feature_view(
#         name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
#     )

#     ts_data = feature_view.get_batch_data(
#         start_time=(fetch_data_from - timedelta(days=1)),
#         end_time=(fetch_data_to + timedelta(days=1)),
#     )
#     ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

#     # Sort data by location and time
#     ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)

#     features = transform_ts_data_info_features(
#         ts_data, window_size=24 * 28, step_size=23
#     )

#     return features


# def load_model_from_registry(version=None):
#     from pathlib import Path

#     import joblib

#     from src.pipeline_utils import (  # Import custom classes/functions
#         TemporalFeatureEngineer,
#         average_rides_last_4_weeks,
#     )

#     project = get_hopsworks_project()
#     model_registry = project.get_model_registry()

#     models = model_registry.get_models(name=config.MODEL_NAME)
#     model = max(models, key=lambda model: model.version)
#     model_dir = model.download()
#     model = joblib.load(Path(model_dir) / "lgb_model.pkl")

#     return model


# def load_metrics_from_registry(version=None):

#     project = get_hopsworks_project()
#     model_registry = project.get_model_registry()

#     models = model_registry.get_models(name=config.MODEL_NAME)
#     model = max(models, key=lambda model: model.version)

#     return model.training_metrics


# def fetch_next_hour_predictions():
#     # Get current UTC time and round up to next hour
#     now = datetime.now(timezone.utc)
#     next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
#     df = fg.read()
#     # Then filter for next hour in the DataFrame
#     df = df[df["pickup_hour"] == next_hour]

#     print(f"Current UTC time: {now}")
#     print(f"Next hour: {next_hour}")
#     print(f"Found {len(df)} records")
#     return df


# def fetch_predictions(hours):
#     current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

#     df = fg.filter((fg.pickup_hour >= current_hour)).read()

#     return df


# def fetch_hourly_rides(hours):
#     current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

#     query = fg.select_all()
#     query = query.filter(fg.pickup_hour >= current_hour)

#     return query.read()


# def fetch_days_data(days):
#     current_date = pd.to_datetime(datetime.now(timezone.utc))
#     fetch_data_from = current_date - timedelta(days=(365 + days))
#     fetch_data_to = current_date 
#     # - timedelta(days=365)
#     print(fetch_data_from, fetch_data_to)
#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

#     query = fg.select_all()
#     # query = query.filter((fg.pickup_hour >= fetch_data_from))
#     df = query.read()
#     cond = (df["pickup_hour"] >= fetch_data_from) & (df["pickup_hour"] <= fetch_data_to)
#     return df[cond]

# from datetime import datetime, timedelta, timezone
# from pathlib import Path

# import hopsworks
# import pandas as pd
# from hsfs.feature_store import FeatureStore

# import src.config as config
# from src.data_utils import transform_ts_data_info_features
# from src.pipeline_utils import (
#     TemporalFeatureEngineer,            # needed for joblib unpickle
#     average_rides_last_4_weeks,         # needed for joblib unpickle
#     ensure_required_lag_features,
#     REQUIRED_LAGS_FOR_AVG_4W,
# )

# # ------------- Hopsworks basics -------------
# def get_hopsworks_project() -> hopsworks.project.Project:
#     return hopsworks.login(
#         project=config.HOPSWORKS_PROJECT_NAME,
#         api_key_value=config.HOPSWORKS_API_KEY,
#     )

# def get_feature_store() -> FeatureStore:
#     return get_hopsworks_project().get_feature_store()

# # ------------- FG-ONLY reader (no FV calls anywhere) -------------
# def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
#     """
#     Read directly from Feature Group only. Never call get_feature_view().
#     Builds 28d (672h) sliding windows with step=1.
#     """
#     fs = get_feature_store()

#     fetch_data_to = current_date - timedelta(hours=1)
#     fetch_data_from = current_date - timedelta(days=29)
#     print(f"[FG-only] Fetching data from {fetch_data_from} to {fetch_data_to}")

#     fg = fs.get_feature_group(
#         name=config.FEATURE_GROUP_NAME,
#         version=config.FEATURE_GROUP_VERSION,
#     )
#     df = fg.select_all().read()

#     required_cols = {"pickup_hour", "pickup_location_id", "rides"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(
#             f"[FG-only] Feature Group missing required columns {sorted(missing)}. "
#             f"Available: {list(df.columns)}"
#         )

#     df = df.copy()
#     df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce", utc=True)
#     ts_data = df[df["pickup_hour"].between(fetch_data_from, fetch_data_to)]
#     if ts_data.empty:
#         raise ValueError("[FG-only] No rows from Feature Group in requested window.")

#     ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

#     features = transform_ts_data_info_features(
#         ts_data,
#         feature_col="rides",
#         window_size=24 * 28,   # 672 hours
#         step_size=1,
#     )
#     if features.empty:
#         raise ValueError("[FG-only] Sliding-window transform produced zero rows; check data continuity.")

#     for c in ["pickup_location_id", "pickup_hour"]:
#         if c not in features.columns:
#             raise ValueError(f"[FG-only] Missing required column after transform: {c}")

#     return features

# # ------------- Predictions helpers -------------
# def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
#     if features is None or features.empty:
#         raise ValueError("get_model_predictions: empty features DataFrame")

#     # ensure weekly-average inputs exist for the FunctionTransformer
#     features = ensure_required_lag_features(
#         features,
#         feature_col="rides",
#         required_lags=REQUIRED_LAGS_FOR_AVG_4W,
#         fill_value=0.0,
#     )

#     if "pickup_hour" not in features.columns:
#         raise ValueError("get_model_predictions: 'pickup_hour' missing before model.predict")

#     if not pd.api.types.is_datetime64_any_dtype(features["pickup_hour"]):
#         features = features.copy()
#         features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], errors="coerce", utc=True)

#     preds = model.predict(features)
#     out = pd.DataFrame({
#         "pickup_location_id": features["pickup_location_id"].values,
#         "predicted_demand": pd.Series(preds).round(0),
#     })
#     return out

# def load_model_from_registry(version=None):
#     import joblib
#     # keep custom transformers importable during unpickle
#     from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks  # noqa: F401
#     mr = get_hopsworks_project().get_model_registry()
#     model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
#     model_dir = model.download()
#     return joblib.load(Path(model_dir) / "lgb_model.pkl")

# def load_metrics_from_registry(version=None):
#     mr = get_hopsworks_project().get_model_registry()
#     model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
#     return model.training_metrics

# # ------------- Convenience fetchers (unchanged) -------------
# def fetch_next_hour_predictions():
#     now = datetime.now(timezone.utc)
#     next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
#     df = fg.read()
#     return df[df["pickup_hour"] == next_hour]

# def fetch_predictions(hours):
#     current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")
#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
#     return fg.filter((fg.pickup_hour >= current_hour)).read()

# src/inference.py
# src/inference.py
# from __future__ import annotations

# from datetime import datetime, timedelta, timezone
# from pathlib import Path

# import hopsworks
# import pandas as pd
# from hsfs.feature_store import FeatureStore

# import src.config as config
# from src.data_utils import transform_ts_data_info_features


# # ---------------- Hopsworks basics ----------------
# def get_hopsworks_project() -> hopsworks.project.Project:
#     return hopsworks.login(
#         project=config.HOPSWORKS_PROJECT_NAME,
#         api_key_value=config.HOPSWORKS_API_KEY,
#     )


# def get_feature_store() -> FeatureStore:
#     return get_hopsworks_project().get_feature_store()


# # ---------------- Feature View helpers ----------------
# def ensure_feature_view(fs: FeatureStore):
#     """
#     Ensure the Feature View exists. If not, create it from the Feature Group.
#     Returns the FeatureView object. Raises if creation isn't possible.
#     """
#     # Try to fetch if present
#     try:
#         return fs.get_feature_view(
#             name=config.FEATURE_VIEW_NAME,
#             version=config.FEATURE_VIEW_VERSION,
#         )
#     except Exception:
#         pass  # Not there → create it

#     # Create from the Feature Group
#     fg = fs.get_feature_group(
#         name=config.FEATURE_GROUP_NAME,
#         version=config.FEATURE_GROUP_VERSION,
#     )
#     q = fg.select_all()
#     fs.create_feature_view(
#         name=config.FEATURE_VIEW_NAME,
#         version=config.FEATURE_VIEW_VERSION,
#         description="Hourly rides per pickup_location_id",
#         query=q,
#         labels=[],  # inference FV (no label column)
#     )
#     fv = fs.get_feature_view(
#         name=config.FEATURE_VIEW_NAME,
#         version=config.FEATURE_VIEW_VERSION,
#     )
#     print(f"[FV] Created {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
#     return fv


# def _read_via_feature_view(
#     fs: FeatureStore,
#     start_ts: pd.Timestamp,
#     end_ts: pd.Timestamp,
# ) -> pd.DataFrame:
#     """
#     Ensure FV exists, read a buffered window, trim to [start_ts, end_ts].
#     pickup_hour returned as UTC tz-aware.
#     """
#     fv = ensure_feature_view(fs)
#     df = fv.get_batch_data(
#         start_time=(start_ts - pd.Timedelta(days=1)),
#         end_time=(end_ts + pd.Timedelta(days=1)),
#     )
#     if "pickup_hour" not in df.columns:
#         raise ValueError("[FV] 'pickup_hour' column missing in Feature View data.")

#     out = df.copy()
#     out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], errors="coerce", utc=True)
#     out = out[out["pickup_hour"].between(start_ts, end_ts)]
#     return out


# # ---------------- Public loader (used by app/pipelines) ----------------
# def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
#     """
#     Read last 29 days up to 1 hour ago via Feature View (creating it if needed),
#     then build 28-day (672h) sliding-window features with step=1.
#     """
#     fs = get_feature_store()

#     ts = pd.Timestamp(current_date)
#     current_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
#     fetch_to = current_utc - pd.Timedelta(hours=1)
#     fetch_from = current_utc - pd.Timedelta(days=29)
#     print(f"[inference] FV window {fetch_from} .. {fetch_to}")

#     # Read time series via FV (auto-create if missing)
#     ts_data = _read_via_feature_view(fs, fetch_from, fetch_to)
#     if ts_data.empty:
#         raise ValueError("[FV] No rows in requested window. Ensure the feature pipeline ingested data.")

#     ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

#     # Build features expected by the model: 28d window, step=1
#     features = transform_ts_data_info_features(
#         ts_data,
#         feature_col="rides",
#         window_size=24 * 28,  # 672 hours
#         step_size=1,
#         tz="America/New_York",
#         fill_missing=True,
#         fill_value=0.0,
#     )
#     if features.empty:
#         raise ValueError("[FV] Sliding-window transform produced zero rows; check data continuity.")

#     # Columns needed downstream
#     for c in ("pickup_location_id", "pickup_hour"):
#         if c not in features.columns:
#             raise ValueError(f"[FV] Missing required column after transform: {c}")

#     return features


# # ---------------- Model + helpers ----------------
# def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
#     if features is None or features.empty:
#         raise ValueError("get_model_predictions: empty features DataFrame")
#     if "pickup_hour" not in features.columns:
#         raise ValueError("get_model_predictions: 'pickup_hour' required")

#     if not pd.api.types.is_datetime64_any_dtype(features["pickup_hour"]):
#         features = features.copy()
#         features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], errors="coerce", utc=True)

#     preds = model.predict(features)
#     return pd.DataFrame(
#         {
#             "pickup_location_id": features["pickup_location_id"].values,
#             "predicted_demand": pd.Series(preds).round(0),
#         }
#     )


# def load_model_from_registry(version=None):
#     import joblib
#     # Ensure custom transformers importable during unpickle
#     from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks  # noqa: F401

#     mr = get_hopsworks_project().get_model_registry()
#     model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
#     model_dir = model.download()
#     return joblib.load(Path(model_dir) / "lgb_model.pkl")


# def load_metrics_from_registry(version=None):
#     mr = get_hopsworks_project().get_model_registry()
#     model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
#     return model.training_metrics


# def fetch_next_hour_predictions():
#     now = datetime.now(timezone.utc)
#     next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
#     df = fg.read()
#     return df[df["pickup_hour"] == next_hour]


# def fetch_predictions(hours: int):
#     current_hour = (pd.Timestamp.now(tz="Etc/UTC") - pd.Timedelta(hours=hours)).floor("h")
#     fs = get_feature_store()
#     fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
#     return fg.filter((fg.pickup_hour >= current_hour)).read()

# src/inference.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features


# ---------------- Hopsworks basics ----------------
def get_hopsworks_project() -> hopsworks.project.Project:
    """
    Login to Hopsworks using env vars from config.py.
    """
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def get_feature_store() -> FeatureStore:
    """
    Return the Feature Store handle for the configured project.
    """
    return get_hopsworks_project().get_feature_store()


# ---------------- Feature View helpers ----------------
def ensure_feature_view(fs: FeatureStore):
    """
    Ensure the Feature View exists and return it.
    Steps:
      1) Try to GET (retrieve).
      2) If missing, CREATE from the configured Feature Group.
      3) GET again (handles race conditions).
    """
    # 1) Try to retrieve
    try:
        return fs.get_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
        )
    except Exception as e_get:
        print(f"[FV] get_feature_view not found yet: {type(e_get).__name__}: {e_get}")

    # 2) Create from Feature Group
    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        query = fg.select_all()
        fs.create_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
            description="Hourly rides per pickup_location_id (auto-created)",
            query=query,
            labels=[],  # inference FV (no label column)
        )
        print(f"[FV] create_feature_view submitted: {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
    except Exception as e_create:
        # Another job may have created it; or permissions may block creation.
        print(f"[FV] create_feature_view warning: {type(e_create).__name__}: {e_create}")

    # 3) Retrieve again (idempotent & handles races)
    fv = fs.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
    )
    print(f"[FV] ready: {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
    return fv


def _read_via_feature_view(
    fs: FeatureStore,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Read a buffered window via Feature View and trim to [start_ts, end_ts].
    Returns tz-aware UTC 'pickup_hour'.
    """
    fv = ensure_feature_view(fs)
    df = fv.get_batch_data(
        start_time=(start_ts - pd.Timedelta(days=1)),
        end_time=(end_ts + pd.Timedelta(days=1)),
    )
    if "pickup_hour" not in df.columns:
        raise ValueError("[FV] 'pickup_hour' column missing in Feature View data.")
    out = df.copy()
    out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], errors="coerce", utc=True)
    out = out[out["pickup_hour"].between(start_ts, end_ts)]
    return out


# ---------------- Public loader (used by app/pipelines) ----------------
def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Read last 29 days up to 1 hour ago via Feature View (creating it if needed),
    then build 28-day (672h) sliding-window features with step=1.
    """
    fs = get_feature_store()

    ts = pd.Timestamp(current_date)
    current_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    fetch_to = current_utc - pd.Timedelta(hours=1)
    fetch_from = current_utc - pd.Timedelta(days=29)
    print(f"[inference] FV window {fetch_from} .. {fetch_to}")

    ts_data = _read_via_feature_view(fs, fetch_from, fetch_to)
    if ts_data.empty:
        raise ValueError(
            "[FV] No rows in requested window. "
            "Ensure the feature pipeline has ingested data into the Feature Group."
        )

    ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

    # Build features the model expects: 28d window, step=1
    features = transform_ts_data_info_features(
        ts_data,
        feature_col="rides",
        window_size=24 * 28,  # 672 hours
        step_size=1,
        tz="America/New_York",
        fill_missing=True,
        fill_value=0.0,
    )
    if features.empty:
        raise ValueError(
            "[features] Sliding-window transform produced zero rows; check data continuity per location."
        )

    # Columns needed downstream
    for c in ("pickup_location_id", "pickup_hour"):
        if c not in features.columns:
            raise ValueError(f"[features] Missing required column after transform: {c}")

    return features


# ---------------- Model + helpers ----------------
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Run model.predict on features and return (pickup_location_id, predicted_demand).
    """
    if features is None or features.empty:
        raise ValueError("get_model_predictions: empty features DataFrame")
    if "pickup_hour" not in features.columns:
        raise ValueError("get_model_predictions: 'pickup_hour' required")

    if not pd.api.types.is_datetime64_any_dtype(features["pickup_hour"]):
        features = features.copy()
        features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], errors="coerce", utc=True)

    preds = model.predict(features)
    return pd.DataFrame(
        {
            "pickup_location_id": features["pickup_location_id"].values,
            "predicted_demand": pd.Series(preds).round(0),
        }
    )


def load_model_from_registry(version=None):
    """
    Download the latest registered model and load it with joblib.
    Ensures your custom transformers are importable during unpickle.
    """
    import joblib

    # Make custom transformers visible for unpickling
    from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks  # noqa: F401

    mr = get_hopsworks_project().get_model_registry()
    model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
    model_dir = model.download()
    return joblib.load(Path(model_dir) / "lgb_model.pkl")


def load_metrics_from_registry(version=None):
    """
    Return training metrics for the latest version of the model.
    """
    mr = get_hopsworks_project().get_model_registry()
    model = max(mr.get_models(name=config.MODEL_NAME), key=lambda m: m.version)
    return model.training_metrics


def fetch_next_hour_predictions():
    """
    Read predictions Feature Group and return only rows whose pickup_hour == next UTC hour.
    """
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()
    return df[df["pickup_hour"] == next_hour]


def fetch_predictions(hours: int):
    """
    Return all predictions where pickup_hour >= (now - hours).
    """
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - pd.Timedelta(hours=hours)).floor("h")
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    return fg.filter((fg.pickup_hour >= current_hour)).read()
