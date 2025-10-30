# import logging
# import os
# import sys
# from datetime import datetime, timedelta, timezone

# import hopsworks
# import pandas as pd

# import src.config as config
# from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,  # Set the logging level
#     format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
#     handlers=[
#         logging.StreamHandler(sys.stdout),  # Output logs to stdout
#     ],
# )
# logger = logging.getLogger(__name__)


# # Step 1: Get the current date and time (timezone-aware)
# current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
# logger.info(f"Current date and time (UTC): {current_date}")

# # Step 2: Define the data fetching range
# fetch_data_to = current_date
# fetch_data_from = current_date - timedelta(days=28)
# logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# # Step 3: Fetch raw data
# logger.info("Fetching raw data...")
# rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
# logger.info(f"Raw data fetched. Number of records: {len(rides)}")

# # Step 4: Transform raw data into time-series data
# logger.info("Transforming raw data into time-series data...")
# ts_data = transform_raw_data_into_ts_data(rides)
# logger.info(
#     f"Transformation complete. Number of records in time-series data: {len(ts_data)}"
# )

# # Step 5: Connect to the Hopsworks project
# logger.info("Connecting to Hopsworks project...")
# project = hopsworks.login(
#     project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
# )
# logger.info("Connected to Hopsworks project.")

# # Step 6: Connect to the feature store
# logger.info("Connecting to the feature store...")
# feature_store = project.get_feature_store()
# logger.info("Connected to the feature store.")

# # Step 7: Connect to or create the feature group
# logger.info(
#     f"Connecting to the feature group: {config.FEATURE_GROUP_NAME} (version {config.FEATURE_GROUP_VERSION})..."
# )
# feature_group = feature_store.get_feature_group(
#     name=config.FEATURE_GROUP_NAME,
#     version=config.FEATURE_GROUP_VERSION,
# )
# logger.info("Feature group ready.")

# # Step 8: Insert data into the feature group
# logger.info("Inserting data into the feature group...")
# feature_group.insert(ts_data, write_options={"wait_for_job": False})
# logger.info("Data insertion completed.")
# from __future__ import annotations

# import logging
# import sys
# from datetime import datetime, timedelta, timezone

# import pandas as pd
# import hopsworks

# import src.config as config
# from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger = logging.getLogger(__name__)

# current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
# logger.info(f"Current date and time (UTC): {current_date}")

# fetch_data_to = current_date
# fetch_data_from = current_date - timedelta(days=28)
# logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# logger.info("Fetching raw data...")
# rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
# logger.info(f"Raw data fetched. Number of records: {len(rides)}")
# if rides.empty:
#     raise ValueError("feature_pipeline: fetched 0 raw records.")

# logger.info("Transforming raw data into time-series data...")
# ts_data = transform_raw_data_into_ts_data(rides)
# logger.info("Transformation complete. Time-series records: %d, columns: %s", len(ts_data), list(ts_data.columns))

# # Coerce dtypes to match FG schema
# def coerce_fg_dtypes(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out["pickup_location_id"] = pd.to_numeric(out["pickup_location_id"], errors="coerce").fillna(0).astype("int32")
#     if not pd.api.types.is_datetime64_any_dtype(out["pickup_hour"]):
#         out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], errors="coerce", utc=True)
#     else:
#         if getattr(out["pickup_hour"].dt, "tz", None) is None:
#             out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], utc=True)
#         else:
#             out["pickup_hour"] = out["pickup_hour"].dt.tz_convert("UTC")
#     out["rides"] = pd.to_numeric(out["rides"], errors="coerce").fillna(0).astype("int32")
#     return out

# ts_data = coerce_fg_dtypes(ts_data)

# logger.info("Connecting to Hopsworks project...")
# project = hopsworks.login(project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY)
# logger.info("Connected to Hopsworks project.")

# logger.info("Connecting to the feature store...")
# feature_store = project.get_feature_store()
# logger.info("Connected to the feature store.")

# logger.info("Connecting to the feature group: %s (version %s)...", config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION)
# feature_group = feature_store.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
# logger.info("Feature group ready.")

# missing = {"pickup_location_id", "pickup_hour", "rides"} - set(ts_data.columns)
# if missing:
#     raise ValueError(f"feature_pipeline: ts_data missing required columns {sorted(missing)}")

# logger.info("Inserting data into the feature group...")
# feature_group.insert(ts_data, write_options={"wait_for_job": False})
# logger.info("Data insertion completed.")

# src/feature_pipeline.py
# from __future__ import annotations

# import logging
# import sys
# from datetime import datetime, timedelta, timezone

# import hopsworks
# import pandas as pd

# import src.config as config
# from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data
# from src.inference import ensure_feature_view  # reuse the same helper


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)

# current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
# log.info(f"Current date and time (UTC): {current_date}")

# fetch_data_to = current_date
# fetch_data_from = current_date - timedelta(days=28)
# log.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# log.info("Fetching raw data...")
# rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
# log.info(f"Raw data fetched. Number of records: {len(rides)}")

# log.info("Transforming raw data into time-series data...")
# ts_data = transform_raw_data_into_ts_data(rides)
# log.info(f"Transformation complete. Time-series records: {len(ts_data)}; columns: {list(ts_data.columns)}")

# log.info("Connecting to Hopsworks project...")
# project = hopsworks.login(project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY)
# log.info("Connected to Hopsworks project.")

# log.info("Connecting to the feature store...")
# fs = project.get_feature_store()
# log.info("Connected to the feature store.")

# log.info(f"Getting FG: {config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION} ...")
# fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)

# log.info("Inserting hourly TS into FG...")
# fg.insert(ts_data, write_options={"wait_for_job": False})
# log.info("Insert submitted.")

# # Ensure FV exists right after ingestion (idempotent)
# try:
#     ensure_feature_view(fs)
#     log.info(f"Feature View ensured: {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
# except Exception as e:
#     log.warning(f"Could not ensure Feature View (will rely on FG in the meantime): {e}")

# src/feature_pipeline.py
# from __future__ import annotations

# import logging
# import sys
# from datetime import datetime, timedelta, timezone

# import hopsworks
# import pandas as pd

# import src.config as config
# from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data
# from src.inference import ensure_feature_view


# # Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)

# # Current hour UTC (ceil)
# current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
# log.info(f"Current date and time (UTC): {current_date}")

# fetch_data_to = current_date
# fetch_data_from = current_date - timedelta(days=28)
# log.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# # 1) Fetch raw trips (your helper does last-year fallback if current month not available)
# log.info("Fetching raw data...")
# rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
# log.info(f"Raw data fetched. Number of records: {len(rides)}")

# # 2) Convert to hourly time series
# log.info("Transforming raw data into time-series data...")
# ts_data = transform_raw_data_into_ts_data(rides)
# log.info(f"Transformation complete. Time-series records: {len(ts_data)}; columns: {list(ts_data.columns)}")

# # 3) Login + Feature Store
# log.info("Connecting to Hopsworks project...")
# project = hopsworks.login(project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY)
# log.info("Connected to Hopsworks project.")

# log.info("Connecting to the feature store...")
# fs = project.get_feature_store()
# log.info("Connected to the feature store.")

# # 4) Insert into Feature Group (WAIT for completion to avoid race with FV creation)
# log.info(f"Getting FG: {config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION} ...")
# fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)

# log.info("Inserting hourly TS into FG...")
# fg.insert(ts_data, write_options={"wait_for_job": True})  # important: True (wait)
# log.info("[FG] Insert completed.")

# # 5) Ensure Feature View exists (idempotent, will show up in UI)
# try:
#     fv = ensure_feature_view(fs)
#     log.info(f"[FV] Ready: {fv.name} v{fv.version}")
# except Exception as e:
#     log.warning(f"Could not ensure Feature View (fix names/permissions): {e}")
#     raise

from __future__ import annotations
import logging, sys
from datetime import datetime, timedelta, timezone
import hopsworks, pandas as pd
import src.config as config
from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data
from src.inference import _ensure_feature_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

current_date = pd.Timestamp(datetime.now(timezone.utc)).ceil("h")
log.info(f"Current date: {current_date}")
fetch_to = current_date
fetch_from = current_date - timedelta(days=28)
log.info(f"Fetching {fetch_from} → {fetch_to}")

log.info("Fetching raw data...")
rides = fetch_batch_raw_data(fetch_from, fetch_to)
log.info(f"Fetched {len(rides)} raw rows")

log.info("Transforming raw → TS data...")
ts_data = transform_raw_data_into_ts_data(rides)
log.info(f"TS data shape: {ts_data.shape}")

project = hopsworks.login(project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)

log.info("Inserting TS data...")
fg.insert(ts_data, write_options={"wait_for_job": True})
log.info("Insert done")

log.info("Ensuring Feature View exists...")
fv = _ensure_feature_view(fs)
log.info(f"Feature View ready: {fv.name if fv else 'None'}")
