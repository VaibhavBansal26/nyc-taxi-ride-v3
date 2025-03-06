import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd

import src.config as config
from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output logs to stdout
    ],
)
logger = logging.getLogger(__name__)


# Step 1: Get the current date and time (timezone-aware)
current_date = pd.to_datetime(datetime.now(timezone.utc)).ceil("h")
logger.info(f"Current date and time (UTC): {current_date}")

# Step 2: Define the data fetching range
fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=28)
logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# Step 3: Fetch raw data
logger.info("Fetching raw data...")
rides = fetch_batch_raw_data(fetch_data_from, fetch_data_to)
logger.info(f"Raw data fetched. Number of records: {len(rides)}")

# Step 4: Transform raw data into time-series data
logger.info("Transforming raw data into time-series data...")
ts_data = transform_raw_data_into_ts_data(rides)
logger.info(
    f"Transformation complete. Number of records in time-series data: {len(ts_data)}"
)

# Step 5: Connect to the Hopsworks project
logger.info("Connecting to Hopsworks project...")
project = hopsworks.login(
    project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
)
logger.info("Connected to Hopsworks project.")

# Step 6: Connect to the feature store
logger.info("Connecting to the feature store...")
feature_store = project.get_feature_store()
logger.info("Connected to the feature store.")

# Step 7: Connect to or create the feature group
logger.info(
    f"Connecting to the feature group: {config.FEATURE_GROUP_NAME} (version {config.FEATURE_GROUP_VERSION})..."
)
feature_group = feature_store.get_feature_group(
    name=config.FEATURE_GROUP_NAME,
    version=config.FEATURE_GROUP_VERSION,
)
logger.info("Feature group ready.")

# Step 8: Insert data into the feature group
logger.info("Inserting data into the feature group...")
feature_group.insert(ts_data, write_options={"wait_for_job": False})
logger.info("Data insertion completed.")
