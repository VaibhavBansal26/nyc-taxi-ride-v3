import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import calendar

# Add the parent directory to the Python path
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
import requests

from src.config import RAW_DATA_DIR


import pyarrow.parquet as pq


def process_zone_data(): 
    zone_path = fetch_zone_data()
    zf = pq.read_table(zone_path)
    zones = zf.to_pandas()
    print("Working on Zone....")
    zones.rename(columns={"LocationID":"pickup_location_id","Zone":"zone"},inplace=True)
    zones.drop(columns=['service_zone','Borough'],inplace=True)
    return zones

def fetch_zone_data() -> str:
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    df = pd.read_csv(url)
    path = Path("..") / "data" / "raw" / "rides_zones.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine='pyarrow', index=False)
    
    print(f"Successfully saved as Parquet: {str(path)}")
    return str(path)

def fetch_raw_trip_data(year: int, month: int) -> Path:
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")


def filter_nyc_taxi_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters NYC Taxi ride data for a specific year and month, removing outliers and invalid records.

    Args:
        rides (pd.DataFrame): DataFrame containing NYC Taxi ride data.
        year (int): Year to filter for.
        month (int): Month to filter for (1-12).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid rides for the specified year and month.

    Raises:
        ValueError: If no valid rides are found or if input parameters are invalid.
    """
    # Validate inputs
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    if not isinstance(year, int) or not isinstance(month, int):
        raise ValueError("Year and month must be integers.")

    # Calculate start and end dates for the specified month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year + (month // 12), month=(month % 12) + 1, day=1)

    # Add a duration column for filtering
    rides["duration"] = rides["tpep_dropoff_datetime"] - rides["tpep_pickup_datetime"]

    # Define filters
    duration_filter = (rides["duration"] > pd.Timedelta(0)) & (
        rides["duration"] <= pd.Timedelta(hours=5)
    )
    total_amount_filter = (rides["total_amount"] > 0) & (
        rides["total_amount"] <= rides["total_amount"].quantile(0.999)
    )
    nyc_location_filter = ~rides["PULocationID"].isin((1, 264, 265))
    date_range_filter = (rides["tpep_pickup_datetime"] >= start_date) & (
        rides["tpep_pickup_datetime"] < end_date
    )

    # Combine all filters
    final_filter = (
        duration_filter & total_amount_filter & nyc_location_filter & date_range_filter
    )

    # Calculate dropped records
    total_records = len(rides)
    valid_records = final_filter.sum()
    records_dropped = total_records - valid_records
    percent_dropped = (records_dropped / total_records) * 100

    print(f"Total records: {total_records:,}")
    print(f"Valid records: {valid_records:,}")
    print(f"Records dropped: {records_dropped:,} ({percent_dropped:.2f}%)")

    # Filter the DataFrame
    validated_rides = rides[final_filter]
    validated_rides = validated_rides[["tpep_pickup_datetime", "PULocationID"]]
    validated_rides.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "PULocationID": "pickup_location_id",
        },
        inplace=True,
    )

    # Verify we have data in the correct time range
    if validated_rides.empty:
        raise ValueError(f"No valid rides found for {year}-{month:02} after filtering.")

    return validated_rides


def load_and_process_taxi_data(
    year: int, months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and process NYC yellow taxi ride data for a specified year and list of months.

    Args:
        year (int): Year to load data for.
        months (Optional[List[int]]): List of months to load. If None, loads all months (1-12).

    Returns:
        pd.DataFrame: Combined and processed ride data for the specified year and months.

    Raises:
        Exception: If no data could be loaded for the specified year and months.
    """

    # Use all months if none are specified
    if months is None:
        months = list(range(1, 13))

    # List to store DataFrames for each month
    monthly_rides = []

    for month in months:
        # Construct the file path
        file_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"

        try:
            # Download the file if it doesn't exist
            if not file_path.exists():
                print(f"Downloading data for {year}-{month:02}...")
                fetch_raw_trip_data(year, month)
                print(f"Successfully downloaded data for {year}-{month:02}.")
            else:
                print(f"File already exists for {year}-{month:02}.")

            # Load the data
            print(f"Loading data for {year}-{month:02}...")
            rides = pd.read_parquet(file_path, engine="pyarrow")

            # Filter and process the data
            rides = filter_nyc_taxi_data(rides, year, month)
            print(f"Successfully processed data for {year}-{month:02}.")

            # zones = process_zone_data()
            # rides = pd.merge(rides,zones, how="left", on="pickup_location_id")

            # rides.drop(columns=["pickup_location_id"], inplace=True)

            # Append the processed DataFrame to the list
            monthly_rides.append(rides)

        except FileNotFoundError:
            print(f"File not found for {year}-{month:02}. Skipping...")
        except Exception as e:
            print(f"Error processing data for {year}-{month:02}: {str(e)}")
            continue

    # Combine all monthly data
    if not monthly_rides:
        raise Exception(
            f"No data could be loaded for the year {year} and specified months: {months}"
        )

    print("Combining all monthly data...")
    combined_rides = pd.concat(monthly_rides, ignore_index=True)
    print("Data loading and processing complete!")

    return combined_rides


# def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
#     """
#     Fills in missing rides for all hours in the range and all unique locations.

#     Parameters:
#     - df: DataFrame with columns [hour_col, location_col, rides_col]
#     - hour_col: Name of the column containing hourly timestamps
#     - location_col: Name of the column containing location IDs
#     - rides_col: Name of the column containing ride counts

#     Returns:
#     - DataFrame with missing hours and locations filled in with 0 rides
#     """
#     # Ensure the hour column is in datetime format
#     df[hour_col] = pd.to_datetime(df[hour_col])

#     # Get the full range of hours (from min to max) with hourly frequency
#     full_hours = pd.date_range(
#         start=df[hour_col].min(), end=df[hour_col].max(), freq="h"
#     )

#     # Get all unique location IDs
#     all_locations = df[location_col].unique()

#     #Create a DataFrame with all combinations of hours and locations
#     full_combinations = pd.DataFrame(
#         [(hour, location,df.loc[df[location_col] == location, "zone"].values[0]) for hour in full_hours for location in all_locations],
#         columns=[hour_col, location_col, "zone"],
#     )
#     # full_combinations = pd.DataFrame(
#     #     [(hour, location) for hour in full_hours for location in all_locations],
#     #     columns=[hour_col, location_col],
#     # )

#     # Merge the original DataFrame with the full combinations DataFrame
#     merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col,"zone"], how="left")

#     # Fill missing rides with 0
#     merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

#     return merged_df

def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    """
    Fills in missing rides for all hours in the range and all unique locations.

    Parameters:
    - df: DataFrame with columns [hour_col, location_col, rides_col, zone_col]
    - hour_col: Name of the column containing hourly timestamps
    - location_col: Name of the column containing location IDs
    - rides_col: Name of the column containing ride counts
    - zone_col: Name of the column containing zone information

    Returns:
    - DataFrame with missing hours and locations filled in with 0 rides
    """
    # Ensure the hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])

    # Get the full range of hours (from min to max) with hourly frequency
    full_hours = pd.date_range(
        start=df[hour_col].min(), end=df[hour_col].max(), freq="h"
    )

    # Get all unique location IDs
    all_locations = df[location_col].unique()

    # Create a DataFrame with all combinations of hours and locations
    full_combinations = pd.DataFrame([(hour, location) for hour in full_hours for location in all_locations],
                                     columns=[hour_col, location_col])

    # Merge the original DataFrame with the full combinations DataFrame
    merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col], how="left")

    # Fill missing ride counts with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    # Fill missing zones using the most frequent zone for each location
    # zone_map = df.groupby(location_col)[zone_col].first().to_dict()
    # merged_df[zone_col] = merged_df[location_col].map(zone_map).fillna("Unknown")

    return merged_df



# def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
#     """
#     Transform raw ride data into time series format.

#     Args:
#         rides: DataFrame with pickup_datetime and location columns

#     Returns:
#         pd.DataFrame: Time series data with filled gaps
#     """
#     # Floor datetime to hour efficiently
#     rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")
#     # .dt.tz_localize("UTC").dt.tz_convert("America/New_York")

#     # Aggregate and fill gaps
#     agg_rides = (
#         rides.groupby(["pickup_hour", "pickup_location_id","zone"])
#         .size()
#         .reset_index(name="rides")
#     )

#     agg_rides_all_slots = (
#         fill_missing_rides_full_range(
#             agg_rides, "pickup_hour", "pickup_location_id", "rides"
#         )
#         .sort_values(["pickup_location_id", "pickup_hour"])
#         .reset_index(drop=True)
#     )

#     # important
#     agg_rides_all_slots = agg_rides_all_slots.astype(
#         {"pickup_location_id": "int16", "rides": "int16","zone":"str"}
#     )
#     return agg_rides_all_slots

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw ride data into time series format.

    Args:
        rides: DataFrame with pickup_datetime and location columns

    Returns:
        pd.DataFrame: Time series data with filled gaps
    """
    # Floor datetime to hour efficiently
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")
    
    # Aggregate and fill gaps
    agg_rides = (
        rides.groupby(["pickup_hour", "pickup_location_id"])
        .size()
        .reset_index(name="rides")
    )

    # Fill missing timestamps, locations, and zones
    agg_rides_all_slots = (
        fill_missing_rides_full_range(
            agg_rides, "pickup_hour", "pickup_location_id", "rides"
        )
        .sort_values(["pickup_location_id", "pickup_hour"])
        .reset_index(drop=True)
    )

    # Important: Ensure correct data types
    agg_rides_all_slots = agg_rides_all_slots.astype(
        {"pickup_location_id": "int16", "rides": "int16"}
    )

    return agg_rides_all_slots


def transform_ts_data_info_features_and_target_loop(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms time series data for all unique location IDs into a tabular format.
    The first `window_size` rows are used as features, and the next row is the target.
    The process slides down by `step_size` rows at a time to create the next set of features and target.
    Feature columns are named based on their hour offsets relative to the target.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data with 'pickup_hour' column.
        feature_col (str): The column name containing the values to use as features and target (default is "rides").
        window_size (int): The number of rows to use as features (default is 12).
        step_size (int): The number of rows to slide the window by (default is 1).

    Returns:
        tuple: (features DataFrame with pickup_hour, targets Series, complete DataFrame)
    """
    # Get all unique location IDs
    location_ids = df["pickup_location_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["pickup_location_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features, and the next value is the target
                features = values[i : i + window_size]
                target = values[i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                # Combine features, target, location_id, and timestamp
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + [
                "target",
                "pickup_location_id",
                "pickup_hour",
            ]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Extract features (including pickup_hour), targets, and keep the complete DataFrame
    features = final_df[feature_columns + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]

    return features, targets


def transform_ts_data_info_features_and_target(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms time series data for all unique location IDs into a tabular format.
    The first `window_size` rows are used as features, and the next row is the target.
    The process slides down by `step_size` rows at a time to create the next set of features and target.
    Feature columns are named based on their hour offsets relative to the target.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data with 'pickup_hour' column.
        feature_col (str): The column name containing the values to use as features and target (default is "rides").
        window_size (int): The number of rows to use as features (default is 12).
        step_size (int): The number of rows to slide the window by (default is 1).

    Returns:
        tuple: (features DataFrame with pickup_hour, targets Series, complete DataFrame)
    """
    # Get all unique location IDs
    location_ids = df["pickup_location_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["pickup_location_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features, and the next value is the target
                features = values[i : i + window_size]
                target = values[i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                # Combine features, target, location_id, and timestamp
                row = np.append(features, [target, location_id,target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + [
                "target",
                "pickup_location_id",
                "pickup_hour",
            ]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Extract features (including pickup_hour), targets, and keep the complete DataFrame
    features = final_df[feature_columns + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]

    return features, targets


def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        cutoff_date (datetime): The date used to split the data into training and testing sets.
        target_column (str): The name of the target column to separate from the features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing target values.
    """
    # Split the data into training and testing sets based on the cutoff date
    train_data = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    # Separate features (X) and target (y) for both training and testing sets
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test


def fetch_batch_raw_data(
    from_date: Union[datetime, str], to_date: Union[datetime, str]
) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e., 1 year).

    Args:
        from_date (datetime or str): The start date for the data batch.
        to_date (datetime or str): The end date for the data batch.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated production data.
    """
    # Convert string inputs to datetime if necessary
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    # Validate input dates
    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError(
            "Both 'from_date' and 'to_date' must be datetime objects or valid ISO format strings."
        )
    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")

    # Shift dates back by 52 weeks (1 year)
    historical_from_date = from_date - timedelta(weeks=52)
    historical_to_date = to_date - timedelta(weeks=52)

    # Load and filter data for the historical period
    rides_from = load_and_process_taxi_data(
        year=historical_from_date.year, months=[historical_from_date.month]
    )
    rides_from = rides_from[
        rides_from.pickup_datetime >= historical_from_date.to_datetime64()
    ]

    if historical_to_date.month != historical_from_date.month:
        rides_to = load_and_process_taxi_data(
            year=historical_to_date.year, months=[historical_to_date.month]
        )
        rides_to = rides_to[
            rides_to.pickup_datetime < historical_to_date.to_datetime64()
        ]
        # Combine the filtered data
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides = rides_from
    # Shift the data forward by 52 weeks to simulate recent data
    rides["pickup_datetime"] += timedelta(weeks=52)

    # Sort the data for consistency
    rides.sort_values(by=["pickup_location_id", "pickup_datetime"], inplace=True)

    return rides


def transform_ts_data_info_features(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms time series data for all unique location IDs into a tabular format.
    The first `window_size` rows are used as features.
    The process slides down by `step_size` rows at a time to create the next set of features.
    Feature columns are named based on their hour offsets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data with 'pickup_hour' column.
        feature_col (str): The column name containing the values to use as features (default is "rides").
        window_size (int): The number of rows to use as features (default is 12).
        step_size (int): The number of rows to slide the window by (default is 1).

    Returns:
        pd.DataFrame: Features DataFrame with pickup_hour and location_id.
    """
    # Get all unique location IDs
    location_ids = df["pickup_location_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["pickup_location_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features
                features = values[i : i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                row = np.append(features, [location_id, target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + ["pickup_location_id", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Return only the features DataFrame
    return final_df
