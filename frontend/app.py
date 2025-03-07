# import sys
# from pathlib import Path

# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)


# import zipfile

# import folium
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import pandas as pd
# import requests
# import streamlit as st
# from branca.colormap import LinearColormap
# from streamlit_folium import st_folium

# from src.config import DATA_DIR
# from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# from src.plot_utils import plot_prediction

# # Add parent directory to Python path


# # Initialize session state for the map
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False


# def visualize_predicted_demand(shapefile_path, predicted_demand):
#     """
#     Visualizes the predicted number of rides on a map of NYC taxi zones.

#     Parameters:
#         shapefile_path (str): Path to the NYC taxi zones shapefile.
#         predicted_demand (dict): A dictionary where keys are taxi zone IDs (or names)
#                                 and values are the predicted number of rides.

#     Returns:
#         None
#     """
#     # Load the shapefile
#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

#     # Ensure the taxi zone IDs in the shapefile match the keys in predicted_demand
#     # Assuming the shapefile has a column 'zone_id' or 'LocationID' for taxi zones
#     if "LocationID" not in gdf.columns:
#         raise ValueError(
#             "Shapefile must contain a 'LocationID' column to match taxi zones."
#         )

#     # Add a new column for predicted rides, defaulting to 0 if no prediction is available
#     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

#     # Plot the map
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     gdf.plot(
#         column="predicted_demand",  # Column to color by
#         cmap="OrRd",  # Color map (e.g., 'OrRd' for orange-red gradient)
#         linewidth=0.8,
#         ax=ax,
#         edgecolor="black",
#         legend=True,
#         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
#     )

#     # Add title and labels
#     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
#     ax.set_axis_off()  # Turn off axis for a cleaner map

#     # Show the plot
#     st.pyplot(fig)


# def create_taxi_map(shapefile_path, prediction_data):
#     """
#     Create an interactive choropleth map of NYC taxi zones with predicted rides
#     """
#     # Load the NYC taxi zones shapefile
#     nyc_zones = gpd.read_file(shapefile_path)

#     # Merge with cleaned column names
#     nyc_zones = nyc_zones.merge(
#         prediction_data[["pickup_location_id", "predicted_demand"]],
#         left_on="LocationID",
#         right_on="pickup_location_id",
#         how="left",
#     )

#     # Fill NaN values with 0 for predicted demand
#     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

#     # Convert to GeoJSON for Folium
#     nyc_zones = nyc_zones.to_crs(epsg=4326)

#     # Create map
#     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

#     # Create color map
#     colormap = LinearColormap(
#         colors=[
#             "#FFEDA0",
#             "#FED976",
#             "#FEB24C",
#             "#FD8D3C",
#             "#FC4E2A",
#             "#E31A1C",
#             "#BD0026",
#         ],
#         vmin=nyc_zones["predicted_demand"].min(),
#         vmax=nyc_zones["predicted_demand"].max(),
#     )

#     colormap.add_to(m)

#     # Define style function
#     def style_function(feature):
#         predicted_demand = feature["properties"].get("predicted_demand", 0)
#         return {
#             "fillColor": colormap(float(predicted_demand)),
#             "color": "black",
#             "weight": 1,
#             "fillOpacity": 0.7,
#         }

#     # Convert GeoDataFrame to GeoJSON
#     zones_json = nyc_zones.to_json()

#     # Add the choropleth layer
#     folium.GeoJson(
#         zones_json,
#         style_function=style_function,
#         tooltip=folium.GeoJsonTooltip(
#             fields=["zone", "predicted_demand"],
#             aliases=["Zone:", "Predicted Demand:"],
#             style=(
#                 "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
#             ),
#         ),
#     ).add_to(m)

#     # Store the map in session state
#     st.session_state.map_obj = m
#     st.session_state.map_created = True
#     return m


# def load_shape_data_file(
#     data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True
# ):
#     """
#     Downloads, extracts, and loads a shapefile as a GeoDataFrame.

#     Parameters:
#         data_dir (str or Path): Directory where the data will be stored.
#         url (str): URL of the shapefile zip file.
#         log (bool): Whether to log progress messages.

#     Returns:
#         GeoDataFrame: The loaded shapefile as a GeoDataFrame.
#     """
#     # Ensure data directory exists
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)

#     # Define file paths
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     # Download the file if it doesn't already exist
#     if not zip_path.exists():
#         if log:
#             print(f"Downloading file from {url}...")
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()  # Raise an HTTPError for bad responses
#             with open(zip_path, "wb") as f:
#                 f.write(response.content)
#             if log:
#                 print(f"File downloaded and saved to {zip_path}")
#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Failed to download file from {url}: {e}")
#     else:
#         if log:
#             print(f"File already exists at {zip_path}, skipping download.")

#     # Extract the zip file if the shapefile doesn't already exist
#     if not shapefile_path.exists():
#         if log:
#             print(f"Extracting files to {extract_path}...")
#         try:
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(extract_path)
#             if log:
#                 print(f"Files extracted to {extract_path}")
#         except zipfile.BadZipFile as e:
#             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
#     else:
#         if log:
#             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

#     # Load and return the shapefile as a GeoDataFrame
#     if log:
#         print(f"Loading shapefile from {shapefile_path}...")
#     try:
#         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#         if log:
#             print("Shapefile successfully loaded.")
#         return gdf
#     except Exception as e:
#         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")


# # st.set_page_config(layout="wide")

# current_date = pd.Timestamp.now(tz="Etc/UTC")
# st.title(f"New York Yellow Taxi Cab Demand Next Hour")
# st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

# progress_bar = st.sidebar.header("Working Progress")
# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4


# with st.spinner(text="Download shape file for taxi zones"):
#     geo_df = load_shape_data_file(DATA_DIR)
#     st.sidebar.write("Shape file was downloaded")
#     progress_bar.progress(1 / N_STEPS)


# with st.spinner(text="Fetching batch of inference data"):
#     features = load_batch_of_features_from_store(current_date)
#     st.sidebar.write("Inference features fetched from the store")
#     progress_bar.progress(2 / N_STEPS)


# with st.spinner(text="Fetching predictions"):
#     predictions = fetch_next_hour_predictions()
#     st.sidebar.write("Model was loaded from the registry")
#     progress_bar.progress(3 / N_STEPS)

# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner(text="Plot predicted rides demand"):
#     # predictions_df = visualize_predicted_demand(
#     #     shapefile_path, predictions["predicted_demand"]
#     # )
#     st.subheader("Taxi Ride Predictions Map")
#     map_obj = create_taxi_map(shapefile_path, predictions)

#     # Display the map
#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

#     # Display data statistics
#     st.subheader("Prediction Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric(
#             "Average Rides",
#             f"{predictions['predicted_demand'].mean():.0f}",
#         )
#     with col2:
#         st.metric(
#             "Maximum Rides",
#             f"{predictions['predicted_demand'].max():.0f}",
#         )
#     with col3:
#         st.metric(
#             "Minimum Rides",
#             f"{predictions['predicted_demand'].min():.0f}",
#         )

#     # Show sample of the data
#     st.sidebar.write("Finished plotting taxi rides demand")
#     progress_bar.progress(4 / N_STEPS)

# st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
# top10 = (
#     predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["pickup_location_id"]
#     .to_list()
# )
# for location_id in top10:
#     fig = plot_prediction(
#         features=features[features["pickup_location_id"] == location_id],
#         prediction=predictions[predictions["pickup_location_id"] == location_id],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# import sys
# from pathlib import Path

# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)

# import zipfile
# import folium
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import pandas as pd
# import requests
# import streamlit as st
# from branca.colormap import LinearColormap
# from streamlit_folium import st_folium

# from src.config import DATA_DIR
# # from src.config import LOOKUP_DIR
# from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# from src.plot_utils import plot_prediction

# # Initialize session state for the map
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False

# def visualize_predicted_demand(shapefile_path, predicted_demand):
#     """
#     Visualizes the predicted number of rides on a map of NYC taxi zones.
#     """
#     # Load the shapefile and convert CRS to WGS84 (lat/lon)
#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

#     if "LocationID" not in gdf.columns:
#         raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")

#     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     gdf.plot(
#         column="predicted_demand",
#         cmap="OrRd",
#         linewidth=0.8,
#         ax=ax,
#         edgecolor="black",
#         legend=True,
#         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
#     )

#     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
#     ax.set_axis_off()
#     st.pyplot(fig)

# def create_taxi_map(shapefile_path, prediction_data):
#     """
#     Create an interactive choropleth map of NYC taxi zones with predicted rides.
#     """
#     nyc_zones = gpd.read_file(shapefile_path)
#     nyc_zones = nyc_zones.merge(
#         prediction_data[["pickup_location_id", "predicted_demand"]],
#         left_on="LocationID",
#         right_on="pickup_location_id",
#         how="left",
#     )
#     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
#     nyc_zones = nyc_zones.to_crs(epsg=4326)

#     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

#     colormap = LinearColormap(
#         colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
#         vmin=nyc_zones["predicted_demand"].min(),
#         vmax=nyc_zones["predicted_demand"].max(),
#     )
#     colormap.add_to(m)

#     def style_function(feature):
#         predicted_demand = feature["properties"].get("predicted_demand", 0)
#         return {
#             "fillColor": colormap(float(predicted_demand)),
#             "color": "black",
#             "weight": 1,
#             "fillOpacity": 0.7,
#         }

#     zones_json = nyc_zones.to_json()
#     folium.GeoJson(
#         zones_json,
#         style_function=style_function,
#         tooltip=folium.GeoJsonTooltip(
#             fields=["zone", "predicted_demand"],
#             aliases=["Zone:", "Predicted Demand:"],
#             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
#         ),
#     ).add_to(m)

#     st.session_state.map_obj = m
#     st.session_state.map_created = True
#     return m

# def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
#     """
#     Downloads, extracts, and loads a shapefile as a GeoDataFrame.
#     """
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     if not zip_path.exists():
#         if log:
#             print(f"Downloading file from {url}...")
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             with open(zip_path, "wb") as f:
#                 f.write(response.content)
#             if log:
#                 print(f"File downloaded and saved to {zip_path}")
#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Failed to download file from {url}: {e}")
#     else:
#         if log:
#             print(f"File already exists at {zip_path}, skipping download.")

#     if not shapefile_path.exists():
#         if log:
#             print(f"Extracting files to {extract_path}...")
#         try:
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(extract_path)
#             if log:
#                 print(f"Files extracted to {extract_path}")
#         except zipfile.BadZipFile as e:
#             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
#     else:
#         if log:
#             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

#     if log:
#         print(f"Loading shapefile from {shapefile_path}...")
#     try:
#         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#         if log:
#             print("Shapefile successfully loaded.")
#         return gdf
#     except Exception as e:
#         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# # Use New York timezone for current date and header
# current_date = pd.Timestamp.now(tz="America/New_York")
# st.title("New York Yellow Taxi Cab Demand Next Hour")
# st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

# progress_bar = st.sidebar.header("Working Progress")
# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4

# with st.spinner(text="Download shape file for taxi zones"):
#     geo_df = load_shape_data_file(DATA_DIR)
#     st.sidebar.write("Shape file was downloaded")
#     progress_bar.progress(1 / N_STEPS)

# with st.spinner(text="Fetching batch of inference data"):
#     features = load_batch_of_features_from_store(current_date)
#     st.sidebar.write("Inference features fetched from the store")
#     progress_bar.progress(2 / N_STEPS)

# with st.spinner(text="Fetching predictions"):
#     predictions = fetch_next_hour_predictions()
#     st.sidebar.write("Model was loaded from the registry")
#     progress_bar.progress(3 / N_STEPS)

# # Convert pickup_hour to New York time if the column exists in predictions
# if "pickup_hour" in predictions.columns:
#     predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # Merge in taxi zone lookup information to add a new "zone" column.
# # Adjust the file path if needed.
# lookup_file = LOOKUP_DIR / "taxi_zone_lookup.csv"
# lookup_df = pd.read_csv(lookup_file)
# # Assuming lookup_df has columns "LocationID" and "zone"
# predictions = predictions.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left"
# )
# # Optionally drop the redundant "LocationID" column after merging:
# predictions.drop(columns=["LocationID"], inplace=True)

# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner(text="Plot predicted rides demand"):
#     st.subheader("Taxi Ride Predictions Map")
#     map_obj = create_taxi_map(shapefile_path, predictions)

#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

#     st.subheader("Prediction Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
#     with col2:
#         st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
#     with col3:
#         st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

#     st.sidebar.write("Finished plotting taxi rides demand")
#     progress_bar.progress(4 / N_STEPS)

# # Display top 10 predictions with the added "zone" column and adjusted pickup hour
# st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

# top10 = (
#     predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["pickup_location_id"]
#     .to_list()
# )
# for location_id in top10:
#     fig = plot_prediction(
#         features=features[features["pickup_location_id"] == location_id],
#         prediction=predictions[predictions["pickup_location_id"] == location_id],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

## version 3
# import sys
# from pathlib import Path
# import zipfile
# import folium
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import pandas as pd
# import requests
# import streamlit as st
# from branca.colormap import LinearColormap
# from streamlit_folium import st_folium

# from src.config import DATA_DIR
# from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# from src.plot_utils import plot_prediction

# # Add parent directory to Python path
# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)

# # Initialize session state for the map
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False

# def visualize_predicted_demand(shapefile_path, predicted_demand):
#     """
#     Visualizes the predicted number of rides on a map of NYC taxi zones.
#     """
#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#     if "LocationID" not in gdf.columns:
#         raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")
#     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     gdf.plot(
#         column="predicted_demand",
#         cmap="OrRd",
#         linewidth=0.8,
#         ax=ax,
#         edgecolor="black",
#         legend=True,
#         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
#     )
#     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
#     ax.set_axis_off()
#     st.pyplot(fig)

# def create_taxi_map(shapefile_path, prediction_data):
#     """
#     Create an interactive choropleth map of NYC taxi zones with predicted rides.
#     """
#     nyc_zones = gpd.read_file(shapefile_path)
#     nyc_zones = nyc_zones.merge(
#         prediction_data[["pickup_location_id", "predicted_demand"]],
#         left_on="LocationID",
#         right_on="pickup_location_id",
#         how="left",
#     )
#     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
#     nyc_zones = nyc_zones.to_crs(epsg=4326)

#     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
#     colormap = LinearColormap(
#         colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
#         vmin=nyc_zones["predicted_demand"].min(),
#         vmax=nyc_zones["predicted_demand"].max(),
#     )
#     colormap.add_to(m)

#     def style_function(feature):
#         predicted_demand = feature["properties"].get("predicted_demand", 0)
#         return {
#             "fillColor": colormap(float(predicted_demand)),
#             "color": "black",
#             "weight": 1,
#             "fillOpacity": 0.7,
#         }

#     zones_json = nyc_zones.to_json()
#     folium.GeoJson(
#         zones_json,
#         style_function=style_function,
#         tooltip=folium.GeoJsonTooltip(
#             fields=["zone", "predicted_demand"],
#             aliases=["Zone:", "Predicted Demand:"],
#             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
#         ),
#     ).add_to(m)

#     st.session_state.map_obj = m
#     st.session_state.map_created = True
#     return m

# def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
#     """
#     Downloads, extracts, and loads a shapefile as a GeoDataFrame.
#     """
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     if not zip_path.exists():
#         if log:
#             print(f"Downloading file from {url}...")
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()
#             with open(zip_path, "wb") as f:
#                 f.write(response.content)
#             if log:
#                 print(f"File downloaded and saved to {zip_path}")
#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Failed to download file from {url}: {e}")
#     else:
#         if log:
#             print(f"File already exists at {zip_path}, skipping download.")

#     if not shapefile_path.exists():
#         if log:
#             print(f"Extracting files to {extract_path}...")
#         try:
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(extract_path)
#             if log:
#                 print(f"Files extracted to {extract_path}")
#         except zipfile.BadZipFile as e:
#             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
#     else:
#         if log:
#             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

#     if log:
#         print(f"Loading shapefile from {shapefile_path}...")
#     try:
#         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#         if log:
#             print("Shapefile successfully loaded.")
#         return gdf
#     except Exception as e:
#         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# # Use New York timezone for current date and header
# current_date = pd.Timestamp.now(tz="America/New_York")
# st.title("New York Yellow Taxi Cab Demand Next Hour")
# st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

# progress_bar = st.sidebar.header("Working Progress")
# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4

# with st.spinner(text="Download shape file for taxi zones"):
#     geo_df = load_shape_data_file(DATA_DIR)
#     st.sidebar.write("Shape file was downloaded")
#     progress_bar.progress(1 / N_STEPS)

# with st.spinner(text="Fetching batch of inference data"):
#     features = load_batch_of_features_from_store(current_date)
#     st.sidebar.write("Inference features fetched from the store")
#     progress_bar.progress(2 / N_STEPS)

# with st.spinner(text="Fetching predictions"):
#     predictions = fetch_next_hour_predictions()
#     st.sidebar.write("Model was loaded from the registry")
#     progress_bar.progress(3 / N_STEPS)

# # Convert pickup_hour to New York time if present in predictions
# if "pickup_hour" in predictions.columns:
#     predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # Use a relative path to load the taxi zone lookup CSV from the project root
# lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
# lookup_df = pd.read_csv(lookup_file)

# # Check for expected columns in the lookup DataFrame.
# # If the column "zone" is not present but "Zone" exists, rename it.
# if "zone" not in lookup_df.columns:
#     if "Zone" in lookup_df.columns:
#         lookup_df.rename(columns={"Zone": "zone"}, inplace=True)
#     else:
#         st.error("The lookup CSV file does not contain a 'zone' column. Found columns: " + ", ".join(lookup_df.columns))
#         st.stop()

# # Merge the lookup data to add a new "zone" column to the predictions.
# predictions = predictions.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left"
# )
# predictions.drop(columns=["LocationID"], inplace=True)

# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner(text="Plot predicted rides demand"):
#     st.subheader("Taxi Ride Predictions Map")
#     map_obj = create_taxi_map(shapefile_path, predictions)

#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

#     st.subheader("Prediction Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
#     with col2:
#         st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
#     with col3:
#         st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

#     st.sidebar.write("Finished plotting taxi rides demand")
#     progress_bar.progress(4 / N_STEPS)

# # Display top 10 predictions with the added "zone" column and adjusted pickup hour
# st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

# top10 = (
#     predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["pickup_location_id"]
#     .to_list()
# )
# for location_id in top10:
#     fig = plot_prediction(
#         features=features[features["pickup_location_id"] == location_id],
#         prediction=predictions[predictions["pickup_location_id"] == location_id],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)


import sys
from pathlib import Path
import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

def visualize_predicted_demand(shapefile_path, predicted_demand):
    """
    Visualizes the predicted number of rides on a map of NYC taxi zones.
    """
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
    if "LocationID" not in gdf.columns:
        raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")
    gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column="predicted_demand",
        cmap="OrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
    )
    ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
    ax.set_axis_off()
    st.pyplot(fig)

def create_taxi_map(shapefile_path, prediction_data):
    """
    Create an interactive choropleth map of NYC taxi zones with predicted rides.
    """
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )
    colormap.add_to(m)

    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    zones_json = nyc_zones.to_json()
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        ),
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    """
    Downloads, extracts, and loads a shapefile as a GeoDataFrame.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# Use New York timezone for current date and header
current_date = pd.Timestamp.now(tz="America/New_York")
st.title("New York Yellow Taxi Cab Demand Next Hour")
st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)

with st.spinner(text="Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

# Convert pickup_hour to New York time if present in predictions
if "pickup_hour" in predictions.columns:
    predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# Use a relative path to load the taxi zone lookup CSV from the project root
lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
lookup_df = pd.read_csv(lookup_file)

# Check for expected columns in the lookup DataFrame.
if "zone" not in lookup_df.columns:
    if "Zone" in lookup_df.columns:
        lookup_df.rename(columns={"Zone": "zone"}, inplace=True)
    else:
        st.error("The lookup CSV file does not contain a 'zone' column. Found columns: " + ", ".join(lookup_df.columns))
        st.stop()

# Merge the lookup data to add a new "zone" column to the predictions.
predictions = predictions.merge(
    lookup_df[["LocationID", "zone"]],
    left_on="pickup_location_id",
    right_on="LocationID",
    how="left"
)
predictions.drop(columns=["LocationID"], inplace=True)

# Add a dropdown to filter predictions by zone.
zone_options = ['All Zones'] + sorted(predictions['zone'].dropna().unique())
selected_zone = st.selectbox("Select a Zone to filter predictions", zone_options)

if selected_zone != "All Zones":
    filtered_predictions = predictions[predictions["zone"] == selected_zone]
else:
    filtered_predictions = predictions

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

with st.spinner(text="Plot predicted rides demand"):
    st.subheader("Taxi Ride Predictions Map")
    # Use filtered predictions for the map as well.
    map_obj = create_taxi_map(shapefile_path, filtered_predictions)

    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
    with col2:
        st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
    with col3:
        st.metric("Minimum Rides", f"{filtered_predictions['predicted_demand'].min():.0f}")

    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(4 / N_STEPS)

# Display top 10 predictions with the added "zone" column and adjusted pickup hour
st.dataframe(filtered_predictions.sort_values("predicted_demand", ascending=False).head(10))

# For each of the top 10 predictions, plot additional details.
top10 = (
    filtered_predictions.sort_values("predicted_demand", ascending=False)
    .head(10)["pickup_location_id"]
    .to_list()
)
for location_id in top10:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=filtered_predictions[filtered_predictions["pickup_location_id"] == location_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
