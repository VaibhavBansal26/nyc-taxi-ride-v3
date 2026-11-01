"""Central configuration: paths, Hopsworks names and runtime flags.

Environment variables are loaded from a ``.env`` file in the repository root
(see ``.env.example``).  On Streamlit Community Cloud the same keys can be set
as *secrets* - top-level secrets are exposed as environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PARENT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PARENT_DIR / ".env")

# --------------------------------------------------------------------------- paths
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PARENT_DIR / "models"

LOCAL_MODEL_PATH = MODELS_DIR / "lgb_model.pkl"
SAMPLE_TS_FILE = SAMPLE_DATA_DIR / "hourly_rides_2024_01.parquet"
ZONE_LOOKUP_FILE = PARENT_DIR / "taxi_zone_lookup.csv"
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TLC_TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet"

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- hopsworks
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY") or None
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME") or None
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST") or None  # only needed for self-hosted clusters

FEATURE_GROUP_NAME = "time_series_hourly_feature_group_v4"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "taxi_hourly_model_prediction_v5"
FEATURE_GROUP_MODEL_PREDICTION_VERSION = 1

# --------------------------------------------------------------------------- modelling
WINDOW_SIZE = 24 * 28  # 672 hourly lags feed the model
LOCAL_TIMEZONE = "America/New_York"
FEATURE_PIPELINE_LOOKBACK_DAYS = 28
INFERENCE_LOOKBACK_DAYS = 29
TRAINING_LOOKBACK_DAYS = 180

# --------------------------------------------------------------------------- app
# "auto"  -> use Hopsworks when an API key is configured and reachable, else demo
# "live"  -> always use Hopsworks (errors are shown)
# "demo"  -> never touch Hopsworks; replay bundled sample data through the local model
APP_MODE = (os.getenv("APP_MODE") or "auto").strip().lower()


def hopsworks_configured() -> bool:
    """True when an API key is present (project name is optional)."""
    return bool(HOPSWORKS_API_KEY)
