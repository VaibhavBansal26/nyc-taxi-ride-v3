# # import sys
# # from pathlib import Path

# # parent_dir = str(Path(__file__).parent.parent)
# # sys.path.append(parent_dir)


# # import zipfile

# # import folium
# # import geopandas as gpd
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import requests
# # import streamlit as st
# # from branca.colormap import LinearColormap
# # from streamlit_folium import st_folium

# # from src.config import DATA_DIR
# # from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# # from src.plot_utils import plot_prediction

# # # Add parent directory to Python path


# # # Initialize session state for the map
# # if "map_created" not in st.session_state:
# #     st.session_state.map_created = False


# # def visualize_predicted_demand(shapefile_path, predicted_demand):
# #     """
# #     Visualizes the predicted number of rides on a map of NYC taxi zones.

# #     Parameters:
# #         shapefile_path (str): Path to the NYC taxi zones shapefile.
# #         predicted_demand (dict): A dictionary where keys are taxi zone IDs (or names)
# #                                 and values are the predicted number of rides.

# #     Returns:
# #         None
# #     """
# #     # Load the shapefile
# #     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

# #     # Ensure the taxi zone IDs in the shapefile match the keys in predicted_demand
# #     # Assuming the shapefile has a column 'zone_id' or 'LocationID' for taxi zones
# #     if "LocationID" not in gdf.columns:
# #         raise ValueError(
# #             "Shapefile must contain a 'LocationID' column to match taxi zones."
# #         )

# #     # Add a new column for predicted rides, defaulting to 0 if no prediction is available
# #     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

# #     # Plot the map
# #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# #     gdf.plot(
# #         column="predicted_demand",  # Column to color by
# #         cmap="OrRd",  # Color map (e.g., 'OrRd' for orange-red gradient)
# #         linewidth=0.8,
# #         ax=ax,
# #         edgecolor="black",
# #         legend=True,
# #         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
# #     )

# #     # Add title and labels
# #     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
# #     ax.set_axis_off()  # Turn off axis for a cleaner map

# #     # Show the plot
# #     st.pyplot(fig)


# # def create_taxi_map(shapefile_path, prediction_data):
# #     """
# #     Create an interactive choropleth map of NYC taxi zones with predicted rides
# #     """
# #     # Load the NYC taxi zones shapefile
# #     nyc_zones = gpd.read_file(shapefile_path)

# #     # Merge with cleaned column names
# #     nyc_zones = nyc_zones.merge(
# #         prediction_data[["pickup_location_id", "predicted_demand"]],
# #         left_on="LocationID",
# #         right_on="pickup_location_id",
# #         how="left",
# #     )

# #     # Fill NaN values with 0 for predicted demand
# #     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

# #     # Convert to GeoJSON for Folium
# #     nyc_zones = nyc_zones.to_crs(epsg=4326)

# #     # Create map
# #     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

# #     # Create color map
# #     colormap = LinearColormap(
# #         colors=[
# #             "#FFEDA0",
# #             "#FED976",
# #             "#FEB24C",
# #             "#FD8D3C",
# #             "#FC4E2A",
# #             "#E31A1C",
# #             "#BD0026",
# #         ],
# #         vmin=nyc_zones["predicted_demand"].min(),
# #         vmax=nyc_zones["predicted_demand"].max(),
# #     )

# #     colormap.add_to(m)

# #     # Define style function
# #     def style_function(feature):
# #         predicted_demand = feature["properties"].get("predicted_demand", 0)
# #         return {
# #             "fillColor": colormap(float(predicted_demand)),
# #             "color": "black",
# #             "weight": 1,
# #             "fillOpacity": 0.7,
# #         }

# #     # Convert GeoDataFrame to GeoJSON
# #     zones_json = nyc_zones.to_json()

# #     # Add the choropleth layer
# #     folium.GeoJson(
# #         zones_json,
# #         style_function=style_function,
# #         tooltip=folium.GeoJsonTooltip(
# #             fields=["zone", "predicted_demand"],
# #             aliases=["Zone:", "Predicted Demand:"],
# #             style=(
# #                 "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
# #             ),
# #         ),
# #     ).add_to(m)

# #     # Store the map in session state
# #     st.session_state.map_obj = m
# #     st.session_state.map_created = True
# #     return m


# # def load_shape_data_file(
# #     data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True
# # ):
# #     """
# #     Downloads, extracts, and loads a shapefile as a GeoDataFrame.

# #     Parameters:
# #         data_dir (str or Path): Directory where the data will be stored.
# #         url (str): URL of the shapefile zip file.
# #         log (bool): Whether to log progress messages.

# #     Returns:
# #         GeoDataFrame: The loaded shapefile as a GeoDataFrame.
# #     """
# #     # Ensure data directory exists
# #     data_dir = Path(data_dir)
# #     data_dir.mkdir(parents=True, exist_ok=True)

# #     # Define file paths
# #     zip_path = data_dir / "taxi_zones.zip"
# #     extract_path = data_dir / "taxi_zones"
# #     shapefile_path = extract_path / "taxi_zones.shp"

# #     # Download the file if it doesn't already exist
# #     if not zip_path.exists():
# #         if log:
# #             print(f"Downloading file from {url}...")
# #         try:
# #             response = requests.get(url, timeout=10)
# #             response.raise_for_status()  # Raise an HTTPError for bad responses
# #             with open(zip_path, "wb") as f:
# #                 f.write(response.content)
# #             if log:
# #                 print(f"File downloaded and saved to {zip_path}")
# #         except requests.exceptions.RequestException as e:
# #             raise Exception(f"Failed to download file from {url}: {e}")
# #     else:
# #         if log:
# #             print(f"File already exists at {zip_path}, skipping download.")

# #     # Extract the zip file if the shapefile doesn't already exist
# #     if not shapefile_path.exists():
# #         if log:
# #             print(f"Extracting files to {extract_path}...")
# #         try:
# #             with zipfile.ZipFile(zip_path, "r") as zip_ref:
# #                 zip_ref.extractall(extract_path)
# #             if log:
# #                 print(f"Files extracted to {extract_path}")
# #         except zipfile.BadZipFile as e:
# #             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
# #     else:
# #         if log:
# #             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

# #     # Load and return the shapefile as a GeoDataFrame
# #     if log:
# #         print(f"Loading shapefile from {shapefile_path}...")
# #     try:
# #         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #         if log:
# #             print("Shapefile successfully loaded.")
# #         return gdf
# #     except Exception as e:
# #         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")


# # # st.set_page_config(layout="wide")

# # current_date = pd.Timestamp.now(tz="Etc/UTC")
# # st.title(f"New York Yellow Taxi Cab Demand Next Hour")
# # st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

# # progress_bar = st.sidebar.header("Working Progress")
# # progress_bar = st.sidebar.progress(0)
# # N_STEPS = 4


# # with st.spinner(text="Download shape file for taxi zones"):
# #     geo_df = load_shape_data_file(DATA_DIR)
# #     st.sidebar.write("Shape file was downloaded")
# #     progress_bar.progress(1 / N_STEPS)


# # with st.spinner(text="Fetching batch of inference data"):
# #     features = load_batch_of_features_from_store(current_date)
# #     st.sidebar.write("Inference features fetched from the store")
# #     progress_bar.progress(2 / N_STEPS)


# # with st.spinner(text="Fetching predictions"):
# #     predictions = fetch_next_hour_predictions()
# #     st.sidebar.write("Model was loaded from the registry")
# #     progress_bar.progress(3 / N_STEPS)

# # shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# # with st.spinner(text="Plot predicted rides demand"):
# #     # predictions_df = visualize_predicted_demand(
# #     #     shapefile_path, predictions["predicted_demand"]
# #     # )
# #     st.subheader("Taxi Ride Predictions Map")
# #     map_obj = create_taxi_map(shapefile_path, predictions)

# #     # Display the map
# #     if st.session_state.map_created:
# #         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

# #     # Display data statistics
# #     st.subheader("Prediction Statistics")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.metric(
# #             "Average Rides",
# #             f"{predictions['predicted_demand'].mean():.0f}",
# #         )
# #     with col2:
# #         st.metric(
# #             "Maximum Rides",
# #             f"{predictions['predicted_demand'].max():.0f}",
# #         )
# #     with col3:
# #         st.metric(
# #             "Minimum Rides",
# #             f"{predictions['predicted_demand'].min():.0f}",
# #         )

# #     # Show sample of the data
# #     st.sidebar.write("Finished plotting taxi rides demand")
# #     progress_bar.progress(4 / N_STEPS)

# # st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
# # top10 = (
# #     predictions.sort_values("predicted_demand", ascending=False)
# #     .head(10)["pickup_location_id"]
# #     .to_list()
# # )
# # for location_id in top10:
# #     fig = plot_prediction(
# #         features=features[features["pickup_location_id"] == location_id],
# #         prediction=predictions[predictions["pickup_location_id"] == location_id],
# #     )
# #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# # import sys
# # from pathlib import Path

# # parent_dir = str(Path(__file__).parent.parent)
# # sys.path.append(parent_dir)

# # import zipfile
# # import folium
# # import geopandas as gpd
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import requests
# # import streamlit as st
# # from branca.colormap import LinearColormap
# # from streamlit_folium import st_folium

# # from src.config import DATA_DIR
# # # from src.config import LOOKUP_DIR
# # from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# # from src.plot_utils import plot_prediction

# # # Initialize session state for the map
# # if "map_created" not in st.session_state:
# #     st.session_state.map_created = False

# # def visualize_predicted_demand(shapefile_path, predicted_demand):
# #     """
# #     Visualizes the predicted number of rides on a map of NYC taxi zones.
# #     """
# #     # Load the shapefile and convert CRS to WGS84 (lat/lon)
# #     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

# #     if "LocationID" not in gdf.columns:
# #         raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")

# #     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

# #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# #     gdf.plot(
# #         column="predicted_demand",
# #         cmap="OrRd",
# #         linewidth=0.8,
# #         ax=ax,
# #         edgecolor="black",
# #         legend=True,
# #         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
# #     )

# #     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
# #     ax.set_axis_off()
# #     st.pyplot(fig)

# # def create_taxi_map(shapefile_path, prediction_data):
# #     """
# #     Create an interactive choropleth map of NYC taxi zones with predicted rides.
# #     """
# #     nyc_zones = gpd.read_file(shapefile_path)
# #     nyc_zones = nyc_zones.merge(
# #         prediction_data[["pickup_location_id", "predicted_demand"]],
# #         left_on="LocationID",
# #         right_on="pickup_location_id",
# #         how="left",
# #     )
# #     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
# #     nyc_zones = nyc_zones.to_crs(epsg=4326)

# #     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

# #     colormap = LinearColormap(
# #         colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
# #         vmin=nyc_zones["predicted_demand"].min(),
# #         vmax=nyc_zones["predicted_demand"].max(),
# #     )
# #     colormap.add_to(m)

# #     def style_function(feature):
# #         predicted_demand = feature["properties"].get("predicted_demand", 0)
# #         return {
# #             "fillColor": colormap(float(predicted_demand)),
# #             "color": "black",
# #             "weight": 1,
# #             "fillOpacity": 0.7,
# #         }

# #     zones_json = nyc_zones.to_json()
# #     folium.GeoJson(
# #         zones_json,
# #         style_function=style_function,
# #         tooltip=folium.GeoJsonTooltip(
# #             fields=["zone", "predicted_demand"],
# #             aliases=["Zone:", "Predicted Demand:"],
# #             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
# #         ),
# #     ).add_to(m)

# #     st.session_state.map_obj = m
# #     st.session_state.map_created = True
# #     return m

# # def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
# #     """
# #     Downloads, extracts, and loads a shapefile as a GeoDataFrame.
# #     """
# #     data_dir = Path(data_dir)
# #     data_dir.mkdir(parents=True, exist_ok=True)
# #     zip_path = data_dir / "taxi_zones.zip"
# #     extract_path = data_dir / "taxi_zones"
# #     shapefile_path = extract_path / "taxi_zones.shp"

# #     if not zip_path.exists():
# #         if log:
# #             print(f"Downloading file from {url}...")
# #         try:
# #             response = requests.get(url, timeout=10)
# #             response.raise_for_status()
# #             with open(zip_path, "wb") as f:
# #                 f.write(response.content)
# #             if log:
# #                 print(f"File downloaded and saved to {zip_path}")
# #         except requests.exceptions.RequestException as e:
# #             raise Exception(f"Failed to download file from {url}: {e}")
# #     else:
# #         if log:
# #             print(f"File already exists at {zip_path}, skipping download.")

# #     if not shapefile_path.exists():
# #         if log:
# #             print(f"Extracting files to {extract_path}...")
# #         try:
# #             with zipfile.ZipFile(zip_path, "r") as zip_ref:
# #                 zip_ref.extractall(extract_path)
# #             if log:
# #                 print(f"Files extracted to {extract_path}")
# #         except zipfile.BadZipFile as e:
# #             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
# #     else:
# #         if log:
# #             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

# #     if log:
# #         print(f"Loading shapefile from {shapefile_path}...")
# #     try:
# #         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #         if log:
# #             print("Shapefile successfully loaded.")
# #         return gdf
# #     except Exception as e:
# #         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# # # Use New York timezone for current date and header
# # current_date = pd.Timestamp.now(tz="America/New_York")
# # st.title("New York Yellow Taxi Cab Demand Next Hour")
# # st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

# # progress_bar = st.sidebar.header("Working Progress")
# # progress_bar = st.sidebar.progress(0)
# # N_STEPS = 4

# # with st.spinner(text="Download shape file for taxi zones"):
# #     geo_df = load_shape_data_file(DATA_DIR)
# #     st.sidebar.write("Shape file was downloaded")
# #     progress_bar.progress(1 / N_STEPS)

# # with st.spinner(text="Fetching batch of inference data"):
# #     features = load_batch_of_features_from_store(current_date)
# #     st.sidebar.write("Inference features fetched from the store")
# #     progress_bar.progress(2 / N_STEPS)

# # with st.spinner(text="Fetching predictions"):
# #     predictions = fetch_next_hour_predictions()
# #     st.sidebar.write("Model was loaded from the registry")
# #     progress_bar.progress(3 / N_STEPS)

# # # Convert pickup_hour to New York time if the column exists in predictions
# # if "pickup_hour" in predictions.columns:
# #     predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # # Merge in taxi zone lookup information to add a new "zone" column.
# # # Adjust the file path if needed.
# # lookup_file = LOOKUP_DIR / "taxi_zone_lookup.csv"
# # lookup_df = pd.read_csv(lookup_file)
# # # Assuming lookup_df has columns "LocationID" and "zone"
# # predictions = predictions.merge(
# #     lookup_df[["LocationID", "zone"]],
# #     left_on="pickup_location_id",
# #     right_on="LocationID",
# #     how="left"
# # )
# # # Optionally drop the redundant "LocationID" column after merging:
# # predictions.drop(columns=["LocationID"], inplace=True)

# # shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# # with st.spinner(text="Plot predicted rides demand"):
# #     st.subheader("Taxi Ride Predictions Map")
# #     map_obj = create_taxi_map(shapefile_path, predictions)

# #     if st.session_state.map_created:
# #         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

# #     st.subheader("Prediction Statistics")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
# #     with col2:
# #         st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
# #     with col3:
# #         st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

# #     st.sidebar.write("Finished plotting taxi rides demand")
# #     progress_bar.progress(4 / N_STEPS)

# # # Display top 10 predictions with the added "zone" column and adjusted pickup hour
# # st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

# # top10 = (
# #     predictions.sort_values("predicted_demand", ascending=False)
# #     .head(10)["pickup_location_id"]
# #     .to_list()
# # )
# # for location_id in top10:
# #     fig = plot_prediction(
# #         features=features[features["pickup_location_id"] == location_id],
# #         prediction=predictions[predictions["pickup_location_id"] == location_id],
# #     )
# #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# ## version 3
# # import sys
# # from pathlib import Path
# # import zipfile
# # import folium
# # import geopandas as gpd
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import requests
# # import streamlit as st
# # from branca.colormap import LinearColormap
# # from streamlit_folium import st_folium

# # from src.config import DATA_DIR
# # from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# # from src.plot_utils import plot_prediction

# # # Add parent directory to Python path
# # parent_dir = str(Path(__file__).parent.parent)
# # sys.path.append(parent_dir)

# # # Initialize session state for the map
# # if "map_created" not in st.session_state:
# #     st.session_state.map_created = False

# # def visualize_predicted_demand(shapefile_path, predicted_demand):
# #     """
# #     Visualizes the predicted number of rides on a map of NYC taxi zones.
# #     """
# #     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #     if "LocationID" not in gdf.columns:
# #         raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")
# #     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)
# #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# #     gdf.plot(
# #         column="predicted_demand",
# #         cmap="OrRd",
# #         linewidth=0.8,
# #         ax=ax,
# #         edgecolor="black",
# #         legend=True,
# #         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
# #     )
# #     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
# #     ax.set_axis_off()
# #     st.pyplot(fig)

# # def create_taxi_map(shapefile_path, prediction_data):
# #     """
# #     Create an interactive choropleth map of NYC taxi zones with predicted rides.
# #     """
# #     nyc_zones = gpd.read_file(shapefile_path)
# #     nyc_zones = nyc_zones.merge(
# #         prediction_data[["pickup_location_id", "predicted_demand"]],
# #         left_on="LocationID",
# #         right_on="pickup_location_id",
# #         how="left",
# #     )
# #     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
# #     nyc_zones = nyc_zones.to_crs(epsg=4326)

# #     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
# #     colormap = LinearColormap(
# #         colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
# #         vmin=nyc_zones["predicted_demand"].min(),
# #         vmax=nyc_zones["predicted_demand"].max(),
# #     )
# #     colormap.add_to(m)

# #     def style_function(feature):
# #         predicted_demand = feature["properties"].get("predicted_demand", 0)
# #         return {
# #             "fillColor": colormap(float(predicted_demand)),
# #             "color": "black",
# #             "weight": 1,
# #             "fillOpacity": 0.7,
# #         }

# #     zones_json = nyc_zones.to_json()
# #     folium.GeoJson(
# #         zones_json,
# #         style_function=style_function,
# #         tooltip=folium.GeoJsonTooltip(
# #             fields=["zone", "predicted_demand"],
# #             aliases=["Zone:", "Predicted Demand:"],
# #             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
# #         ),
# #     ).add_to(m)

# #     st.session_state.map_obj = m
# #     st.session_state.map_created = True
# #     return m

# # def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
# #     """
# #     Downloads, extracts, and loads a shapefile as a GeoDataFrame.
# #     """
# #     data_dir = Path(data_dir)
# #     data_dir.mkdir(parents=True, exist_ok=True)
# #     zip_path = data_dir / "taxi_zones.zip"
# #     extract_path = data_dir / "taxi_zones"
# #     shapefile_path = extract_path / "taxi_zones.shp"

# #     if not zip_path.exists():
# #         if log:
# #             print(f"Downloading file from {url}...")
# #         try:
# #             response = requests.get(url, timeout=10)
# #             response.raise_for_status()
# #             with open(zip_path, "wb") as f:
# #                 f.write(response.content)
# #             if log:
# #                 print(f"File downloaded and saved to {zip_path}")
# #         except requests.exceptions.RequestException as e:
# #             raise Exception(f"Failed to download file from {url}: {e}")
# #     else:
# #         if log:
# #             print(f"File already exists at {zip_path}, skipping download.")

# #     if not shapefile_path.exists():
# #         if log:
# #             print(f"Extracting files to {extract_path}...")
# #         try:
# #             with zipfile.ZipFile(zip_path, "r") as zip_ref:
# #                 zip_ref.extractall(extract_path)
# #             if log:
# #                 print(f"Files extracted to {extract_path}")
# #         except zipfile.BadZipFile as e:
# #             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
# #     else:
# #         if log:
# #             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

# #     if log:
# #         print(f"Loading shapefile from {shapefile_path}...")
# #     try:
# #         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #         if log:
# #             print("Shapefile successfully loaded.")
# #         return gdf
# #     except Exception as e:
# #         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# # # Use New York timezone for current date and header
# # current_date = pd.Timestamp.now(tz="America/New_York")
# # st.title("New York Yellow Taxi Cab Demand Next Hour")
# # st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

# # progress_bar = st.sidebar.header("Working Progress")
# # progress_bar = st.sidebar.progress(0)
# # N_STEPS = 4

# # with st.spinner(text="Download shape file for taxi zones"):
# #     geo_df = load_shape_data_file(DATA_DIR)
# #     st.sidebar.write("Shape file was downloaded")
# #     progress_bar.progress(1 / N_STEPS)

# # with st.spinner(text="Fetching batch of inference data"):
# #     features = load_batch_of_features_from_store(current_date)
# #     st.sidebar.write("Inference features fetched from the store")
# #     progress_bar.progress(2 / N_STEPS)

# # with st.spinner(text="Fetching predictions"):
# #     predictions = fetch_next_hour_predictions()
# #     st.sidebar.write("Model was loaded from the registry")
# #     progress_bar.progress(3 / N_STEPS)

# # # Convert pickup_hour to New York time if present in predictions
# # if "pickup_hour" in predictions.columns:
# #     predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # # Use a relative path to load the taxi zone lookup CSV from the project root
# # lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
# # lookup_df = pd.read_csv(lookup_file)

# # # Check for expected columns in the lookup DataFrame.
# # # If the column "zone" is not present but "Zone" exists, rename it.
# # if "zone" not in lookup_df.columns:
# #     if "Zone" in lookup_df.columns:
# #         lookup_df.rename(columns={"Zone": "zone"}, inplace=True)
# #     else:
# #         st.error("The lookup CSV file does not contain a 'zone' column. Found columns: " + ", ".join(lookup_df.columns))
# #         st.stop()

# # # Merge the lookup data to add a new "zone" column to the predictions.
# # predictions = predictions.merge(
# #     lookup_df[["LocationID", "zone"]],
# #     left_on="pickup_location_id",
# #     right_on="LocationID",
# #     how="left"
# # )
# # predictions.drop(columns=["LocationID"], inplace=True)

# # shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# # with st.spinner(text="Plot predicted rides demand"):
# #     st.subheader("Taxi Ride Predictions Map")
# #     map_obj = create_taxi_map(shapefile_path, predictions)

# #     if st.session_state.map_created:
# #         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

# #     st.subheader("Prediction Statistics")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
# #     with col2:
# #         st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
# #     with col3:
# #         st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

# #     st.sidebar.write("Finished plotting taxi rides demand")
# #     progress_bar.progress(4 / N_STEPS)

# # # Display top 10 predictions with the added "zone" column and adjusted pickup hour
# # st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

# # top10 = (
# #     predictions.sort_values("predicted_demand", ascending=False)
# #     .head(10)["pickup_location_id"]
# #     .to_list()
# # )
# # for location_id in top10:
# #     fig = plot_prediction(
# #         features=features[features["pickup_location_id"] == location_id],
# #         prediction=predictions[predictions["pickup_location_id"] == location_id],
# #     )
# #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)


# # import sys
# # from pathlib import Path
# # import zipfile
# # import folium
# # import geopandas as gpd
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import requests
# # import streamlit as st
# # from branca.colormap import LinearColormap
# # from streamlit_folium import st_folium

# # from src.config import DATA_DIR
# # from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
# # from src.plot_utils import plot_prediction

# # # Add parent directory to Python path
# # parent_dir = str(Path(__file__).parent.parent)
# # sys.path.append(parent_dir)

# # # Initialize session state for the map
# # if "map_created" not in st.session_state:
# #     st.session_state.map_created = False

# # def visualize_predicted_demand(shapefile_path, predicted_demand):
# #     """
# #     Visualizes the predicted number of rides on a map of NYC taxi zones.
# #     """
# #     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #     if "LocationID" not in gdf.columns:
# #         raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")
# #     gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)
# #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# #     gdf.plot(
# #         column="predicted_demand",
# #         cmap="OrRd",
# #         linewidth=0.8,
# #         ax=ax,
# #         edgecolor="black",
# #         legend=True,
# #         legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
# #     )
# #     ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
# #     ax.set_axis_off()
# #     st.pyplot(fig)

# # def create_taxi_map(shapefile_path, prediction_data):
# #     """
# #     Create an interactive choropleth map of NYC taxi zones with predicted rides.
# #     """
# #     nyc_zones = gpd.read_file(shapefile_path)
# #     nyc_zones = nyc_zones.merge(
# #         prediction_data[["pickup_location_id", "predicted_demand"]],
# #         left_on="LocationID",
# #         right_on="pickup_location_id",
# #         how="left",
# #     )
# #     nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
# #     nyc_zones = nyc_zones.to_crs(epsg=4326)

# #     m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
# #     colormap = LinearColormap(
# #         colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
# #         vmin=nyc_zones["predicted_demand"].min(),
# #         vmax=nyc_zones["predicted_demand"].max(),
# #     )
# #     colormap.add_to(m)

# #     def style_function(feature):
# #         predicted_demand = feature["properties"].get("predicted_demand", 0)
# #         return {
# #             "fillColor": colormap(float(predicted_demand)),
# #             "color": "black",
# #             "weight": 1,
# #             "fillOpacity": 0.7,
# #         }

# #     zones_json = nyc_zones.to_json()
# #     folium.GeoJson(
# #         zones_json,
# #         style_function=style_function,
# #         tooltip=folium.GeoJsonTooltip(
# #             fields=["zone", "predicted_demand"],
# #             aliases=["Zone:", "Predicted Demand:"],
# #             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
# #         ),
# #     ).add_to(m)

# #     st.session_state.map_obj = m
# #     st.session_state.map_created = True
# #     return m

# # def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
# #     """
# #     Downloads, extracts, and loads a shapefile as a GeoDataFrame.
# #     """
# #     data_dir = Path(data_dir)
# #     data_dir.mkdir(parents=True, exist_ok=True)
# #     zip_path = data_dir / "taxi_zones.zip"
# #     extract_path = data_dir / "taxi_zones"
# #     shapefile_path = extract_path / "taxi_zones.shp"

# #     if not zip_path.exists():
# #         if log:
# #             print(f"Downloading file from {url}...")
# #         try:
# #             response = requests.get(url, timeout=10)
# #             response.raise_for_status()
# #             with open(zip_path, "wb") as f:
# #                 f.write(response.content)
# #             if log:
# #                 print(f"File downloaded and saved to {zip_path}")
# #         except requests.exceptions.RequestException as e:
# #             raise Exception(f"Failed to download file from {url}: {e}")
# #     else:
# #         if log:
# #             print(f"File already exists at {zip_path}, skipping download.")

# #     if not shapefile_path.exists():
# #         if log:
# #             print(f"Extracting files to {extract_path}...")
# #         try:
# #             with zipfile.ZipFile(zip_path, "r") as zip_ref:
# #                 zip_ref.extractall(extract_path)
# #             if log:
# #                 print(f"Files extracted to {extract_path}")
# #         except zipfile.BadZipFile as e:
# #             raise Exception(f"Failed to extract zip file {zip_path}: {e}")
# #     else:
# #         if log:
# #             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

# #     if log:
# #         print(f"Loading shapefile from {shapefile_path}...")
# #     try:
# #         gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
# #         if log:
# #             print("Shapefile successfully loaded.")
# #         return gdf
# #     except Exception as e:
# #         raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# # # Use New York timezone for current date and header
# # current_date = pd.Timestamp.now(tz="America/New_York")
# # st.title("New York Yellow Taxi Cab Demand Next Hour")
# # st.header(current_date.strftime("%Y-%m-%d %H:%M:%S"))

# # progress_bar = st.sidebar.header("Working Progress")
# # progress_bar = st.sidebar.progress(0)
# # N_STEPS = 4

# # with st.spinner(text="Download shape file for taxi zones"):
# #     geo_df = load_shape_data_file(DATA_DIR)
# #     st.sidebar.write("Shape file was downloaded")
# #     progress_bar.progress(1 / N_STEPS)

# # with st.spinner(text="Fetching batch of inference data"):
# #     features = load_batch_of_features_from_store(current_date)
# #     st.sidebar.write("Inference features fetched from the store")
# #     progress_bar.progress(2 / N_STEPS)

# # with st.spinner(text="Fetching predictions"):
# #     predictions = fetch_next_hour_predictions()
# #     st.sidebar.write("Model was loaded from the registry")
# #     progress_bar.progress(3 / N_STEPS)

# # # Convert pickup_hour to New York time if present in predictions
# # if "pickup_hour" in predictions.columns:
# #     predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # # Use a relative path to load the taxi zone lookup CSV from the project root
# # lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
# # lookup_df = pd.read_csv(lookup_file)

# # # Check for expected columns in the lookup DataFrame.
# # if "zone" not in lookup_df.columns:
# #     if "Zone" in lookup_df.columns:
# #         lookup_df.rename(columns={"Zone": "zone"}, inplace=True)
# #     else:
# #         st.error("The lookup CSV file does not contain a 'zone' column. Found columns: " + ", ".join(lookup_df.columns))
# #         st.stop()

# # # Merge the lookup data to add a new "zone" column to the predictions.
# # predictions = predictions.merge(
# #     lookup_df[["LocationID", "zone"]],
# #     left_on="pickup_location_id",
# #     right_on="LocationID",
# #     how="left"
# # )
# # predictions.drop(columns=["LocationID"], inplace=True)

# # # Add a dropdown to filter predictions by zone.
# # zone_options = ['All Zones'] + sorted(predictions['zone'].dropna().unique())
# # selected_zone = st.selectbox("Select a Zone to filter predictions", zone_options)

# # if selected_zone != "All Zones":
# #     filtered_predictions = predictions[predictions["zone"] == selected_zone]
# # else:
# #     filtered_predictions = predictions

# # shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# # with st.spinner(text="Plot predicted rides demand"):
# #     st.subheader("Taxi Ride Predictions Map")
# #     # Use filtered predictions for the map as well.
# #     map_obj = create_taxi_map(shapefile_path, filtered_predictions)

# #     if st.session_state.map_created:
# #         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

# #     st.subheader("Prediction Statistics")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
# #     with col2:
# #         st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
# #     with col3:
# #         st.metric("Minimum Rides", f"{filtered_predictions['predicted_demand'].min():.0f}")

# #     st.sidebar.write("Finished plotting taxi rides demand")
# #     progress_bar.progress(4 / N_STEPS)

# # # Display top 10 predictions with the added "zone" column and adjusted pickup hour
# # st.dataframe(filtered_predictions.sort_values("predicted_demand", ascending=False).head(10))

# # # For each of the top 10 predictions, plot additional details.
# # top10 = (
# #     filtered_predictions.sort_values("predicted_demand", ascending=False)
# #     .head(10)["pickup_location_id"]
# #     .to_list()
# # )
# # for location_id in top10:
# #     fig = plot_prediction(
# #         features=features[features["pickup_location_id"] == location_id],
# #         prediction=filtered_predictions[filtered_predictions["pickup_location_id"] == location_id],
# #     )
# #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)


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
#     # Fetch predictions using our enhanced data fetching (with expanded time range and fallback)
#     predictions = fetch_next_hour_predictions()
#     if predictions.empty:
#         st.error("No prediction records found for the current time window. Please try again later.")
#         st.stop()
#     # Ensure that predicted_demand is never negative.
#     predictions['predicted_demand'] = predictions['predicted_demand'].clip(lower=0)
#     # Convert pickup_hour in predictions to New York time, if it exists.
#     if "pickup_hour" in predictions.columns:
#         predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     st.sidebar.write("Predictions fetched successfully")
#     progress_bar.progress(3 / N_STEPS)

# # Load the lookup CSV and ensure it has a "zone" column.
# lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
# lookup_df = pd.read_csv(lookup_file)
# if "zone" not in lookup_df.columns:
#     if "Zone" in lookup_df.columns:
#         lookup_df.rename(columns={"Zone": "zone"}, inplace=True)
#     else:
#         st.error("The lookup CSV file does not contain a 'zone' column. Found columns: " + ", ".join(lookup_df.columns))
#         st.stop()

# # Merge the lookup data into predictions to add the "zone" column.
# predictions = predictions.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left"
# )
# predictions.drop(columns=["LocationID"], inplace=True)

# # Convert pickup_hour in features to New York time, if it exists.
# if "pickup_hour" in features.columns:
#     features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True).dt.tz_convert("America/New_York")

# # Merge lookup data into features so that we have a "zone" column.
# features = features.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left"
# )
# features.drop(columns=["LocationID"], inplace=True)

# # Add a dropdown to filter predictions by zone.
# zone_options = ['All Zones'] + sorted(predictions['zone'].dropna().unique())
# selected_zone = st.selectbox("Select a Zone to filter predictions", zone_options)

# if selected_zone != "All Zones":
#     filtered_predictions = predictions[predictions["zone"] == selected_zone]
# else:
#     filtered_predictions = predictions

# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner(text="Plot predicted rides demand"):
#     st.subheader("Taxi Ride Predictions Map")
#     # Use filtered predictions for the map.
#     map_obj = create_taxi_map(shapefile_path, filtered_predictions)
#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])
#     st.subheader("Prediction Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
#     with col2:
#         st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
#     with col3:
#         min_rides = filtered_predictions['predicted_demand'].min()
#         if min_rides < 0:
#             min_rides = 0
#         st.metric("Minimum Rides", f"{min_rides:.0f}")
#     st.sidebar.write("Finished plotting taxi rides demand")
#     progress_bar.progress(4 / N_STEPS)

# # Display top 10 predictions with the added "zone" column.
# st.dataframe(filtered_predictions.sort_values("predicted_demand", ascending=False).head(10))

# # Update plotting: iterate over top 10 zones instead of location IDs.
# top10_zones = (
#     filtered_predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["zone"]
#     .to_list()
# )
# for zone in top10_zones:
#     fig = plot_prediction(
#         features=features[features["zone"] == zone],
#         prediction=filtered_predictions[filtered_predictions["zone"] == zone],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)
# frontend/app.py
# import sys
# from pathlib import Path
# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)

# # Force Streamlit to use latest src.inference (avoid old cached import)
# from importlib import reload as _reload
# import src.inference as _inf
# _inf = _reload(_inf)

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
# from src.plot_utils import plot_prediction
# from src.inference import fetch_next_hour_predictions  # this is fine to import directly


# # Initialize session state for the map
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False


# def create_taxi_map(shapefile_path, prediction_data):
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
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     if not zip_path.exists():
#         if log:
#             print(f"Downloading file from {url}...")
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         with open(zip_path, "wb") as f:
#             f.write(response.content)
#         if log:
#             print(f"File downloaded and saved to {zip_path}")
#     else:
#         if log:
#             print(f"File already exists at {zip_path}, skipping download.")

#     if not shapefile_path.exists():
#         if log:
#             print(f"Extracting files to {extract_path}...")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(extract_path)
#         if log:
#             print(f"Files extracted to {extract_path}")
#     else:
#         if log:
#             print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

#     if log:
#         print(f"Loading shapefile from {shapefile_path}...")
#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#     if log:
#         print("Shapefile successfully loaded.")
#     return gdf


# # Use New York timezone for header only (loader uses UTC internally)
# current_date_header = pd.Timestamp.now(tz="America/New_York")
# st.title("New York Yellow Taxi Cab Demand Next Hour")
# st.header(current_date_header.strftime("%Y-%m-%d %H:%M:%S"))

# progress_bar = st.sidebar.header("Working Progress")
# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4

# with st.spinner(text="Download shape file for taxi zones"):
#     geo_df = load_shape_data_file(DATA_DIR)
#     st.sidebar.write("Shape file was downloaded")
#     progress_bar.progress(1 / N_STEPS)

# with st.spinner(text="Fetching batch of inference data"):
#     # Call robust loader from the freshly reloaded module
#     current_utc = pd.Timestamp.now(tz="Etc/UTC")
#     try:
#         features = _inf.load_batch_of_features_from_store(current_utc)
#     except Exception as e:
#         st.error(
#             "Could not fetch features from Hopsworks (Feature View + fallback).\n\n"
#             "Common causes: FV not created yet, permissions, or empty time window.\n\n"
#             f"Details: {type(e).__name__}: {e}"
#         )
#         st.stop()
#     st.sidebar.write("Inference features fetched from the store")
#     progress_bar.progress(2 / N_STEPS)

# with st.spinner(text="Fetching predictions"):
#     predictions = fetch_next_hour_predictions()
#     if predictions.empty:
#         st.error("No prediction records found for the current time window. Please try again later.")
#         st.stop()
#     predictions["predicted_demand"] = predictions["predicted_demand"].clip(lower=0)
#     if "pickup_hour" in predictions.columns:
#         predictions["pickup_hour"] = pd.to_datetime(predictions["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     st.sidebar.write("Predictions fetched successfully")
#     progress_bar.progress(3 / N_STEPS)

# # Add zone names to predictions and features
# lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
# lookup_df = pd.read_csv(lookup_file)
# if "zone" not in lookup_df.columns and "Zone" in lookup_df.columns:
#     lookup_df = lookup_df.rename(columns={"Zone": "zone"})

# if "zone" not in lookup_df.columns:
#     st.error("The lookup CSV does not contain a 'zone' column.")
#     st.stop()

# predictions = predictions.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left",
# ).drop(columns=["LocationID"])

# # Convert features' pickup_hour to NY time and add zone
# if "pickup_hour" in features.columns:
#     features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
# features = features.merge(
#     lookup_df[["LocationID", "zone"]],
#     left_on="pickup_location_id",
#     right_on="LocationID",
#     how="left",
# ).drop(columns=["LocationID"])

# # Filter UI
# zone_options = ["All Zones"] + sorted(predictions["zone"].dropna().unique())
# selected_zone = st.selectbox("Select a Zone to filter predictions", zone_options)
# filtered_predictions = predictions if selected_zone == "All Zones" else predictions[predictions["zone"] == selected_zone]

# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner(text="Plot predicted rides demand"):
#     st.subheader("Taxi Ride Predictions Map")
#     create_taxi_map(shapefile_path, filtered_predictions)
#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

#     st.subheader("Prediction Statistics")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
#     with col2:
#         st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
#     with col3:
#         min_rides = max(filtered_predictions['predicted_demand'].min(), 0)
#         st.metric("Minimum Rides", f"{min_rides:.0f}")

#     st.sidebar.write("Finished plotting taxi rides demand")
#     progress_bar.progress(4 / N_STEPS)

# st.dataframe(filtered_predictions.sort_values("predicted_demand", ascending=False).head(10))

# # Optional: per-zone history charts
# top10_zones = (
#     filtered_predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["zone"]
#     .to_list()
# )

# frontend/app.py

# import sys
# from pathlib import Path

# # Make src importable
# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)

# # Avoid old cached version on Streamlit Cloud
# from importlib import reload as _reload
# import src.inference as _inf
# _inf = _reload(_inf)

# import zipfile
# import folium
# import geopandas as gpd
# import pandas as pd
# import requests
# import streamlit as st
# from branca.colormap import LinearColormap
# from streamlit_folium import st_folium

# from src.config import DATA_DIR
# from src.plot_utils import plot_prediction
# from src.inference import fetch_next_hour_predictions  # next-hour fetch

# # ---------- Page + simple theme ----------
# st.set_page_config(
#     page_title="NYC Taxi Demand — Next Hour",
#     page_icon="🚕",
#     layout="wide",
# )

# # ---------- Hero header with NYC taxi image ----------
# BANNER_URL = (
#     "https://images.unsplash.com/photo-1483721168571-c0895f4432c7?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bmV3JTIweW9yayUyMHRheGl8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=900"
# )

# hero_left, hero_right = st.columns([1.2, 1], gap="large")
# with hero_left:
#     st.markdown("### 🚕 NYC Taxi Demand")
#     st.markdown(
#         "Predicting **next-hour rides** per taxi zone. "
#         "If the newest batch isn’t ready yet, we’ll show the latest available hour."
#     )
#     now_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S")
#     st.caption(f"Rendered at **{now_ny}** (America/New_York)")
# with hero_right:
#     st.image(BANNER_URL, use_container_width=True)

# st.divider()

# # ---------- Session/state ----------
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False

# # ---------- Helpers ----------
# def create_taxi_map(shapefile_path: Path, prediction_data: pd.DataFrame):
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
#         vmin=float(nyc_zones["predicted_demand"].min()) if len(nyc_zones) else 0.0,
#         vmax=float(nyc_zones["predicted_demand"].max()) if len(nyc_zones) else 1.0,
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
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     if not zip_path.exists():
#         if log:
#             st.write("⬇️ Downloading NYC taxi zones…")
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()
#         zip_path.write_bytes(response.content)

#     if not shapefile_path.exists():
#         if log:
#             st.write("📦 Extracting shapes…")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(extract_path)

#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#     return gdf


# def _attach_zone_names(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
#     if df.empty:
#         return df
#     df = df.copy()
#     df["predicted_demand"] = pd.to_numeric(df["predicted_demand"], errors="coerce").clip(lower=0)
#     if "pickup_hour" in df.columns:
#         df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     df = df.merge(
#         lookup_df[["LocationID", "zone"]],
#         left_on="pickup_location_id",
#         right_on="LocationID",
#         how="left",
#     ).drop(columns=["LocationID"])
#     return df


# # ---------- Sidebar progress ----------
# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4

# with st.spinner("Download shape file for taxi zones"):
#     _ = load_shape_data_file(DATA_DIR)
#     progress_bar.progress(1 / N_STEPS)

# with st.spinner("Fetching features from store (FV + fallback)"):
#     # robust loader from src.inference (auto-creates FV, falls back to FG)
#     current_utc = pd.Timestamp.now(tz="Etc/UTC")
#     try:
#         features = _inf.load_batch_of_features_from_store(current_utc)
#     except Exception as e:
#         st.error(
#             "Could not fetch features from Hopsworks (Feature View + FG fallback).\n\n"
#             "Common causes: FV not created yet, permissions, or empty time window.\n\n"
#             f"Details: {type(e).__name__}: {e}"
#         )
#         st.stop()
#     progress_bar.progress(2 / N_STEPS)

# # ---------- Predictions with graceful fallback ----------
# with st.spinner("Fetching predictions"):
#     # Try next-hour first
#     predictions = fetch_next_hour_predictions()
#     used_fallback = False

#     if predictions.empty:
#         # Show latest available (current/previous hour) so UI never looks empty
#         st.info("Next-hour predictions not ready — showing the latest available hour.", icon="⏳")
#         history = _inf.fetch_predictions(hours=6)  # look back up to 6 hours
#         if history.empty:
#             st.error("No historical predictions found in the last 6 hours.")
#             st.stop()
#         latest_ts = pd.to_datetime(history["pickup_hour"], utc=True).max()
#         predictions = history[history["pickup_hour"] == latest_ts].copy()
#         used_fallback = True

#     # Attach zone names
#     lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
#     lookup_df = pd.read_csv(lookup_file)
#     if "zone" not in lookup_df.columns and "Zone" in lookup_df.columns:
#         lookup_df.rename(columns={"Zone": "zone"}, inplace=True)

#     predictions = _attach_zone_names(predictions, lookup_df)

#     # Convert features pickup_hour to NY time and attach zones (for charts)
#     if "pickup_hour" in features.columns:
#         features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     features = features.merge(
#         lookup_df[["LocationID", "zone"]],
#         left_on="pickup_location_id",
#         right_on="LocationID",
#         how="left",
#     ).drop(columns=["LocationID"])

#     progress_bar.progress(3 / N_STEPS)

# # ---------- Filters ----------
# zones = ["All Zones"] + sorted(predictions["zone"].dropna().unique().tolist())
# selected_zone = st.selectbox("Filter by Zone", zones, index=0)
# filtered_predictions = predictions if selected_zone == "All Zones" else predictions[predictions["zone"] == selected_zone]

# st.caption(
#     "Showing **next hour**" if not used_fallback else "Showing **latest available hour** (next hour not ready yet)"
# )

# # ---------- Map + KPIs ----------
# shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# with st.spinner("Rendering map"):
#     st.subheader("Choropleth Map")
#     create_taxi_map(shapefile_path, filtered_predictions)
#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=1100, height=560, returned_objects=[])

# k1, k2, k3 = st.columns(3)
# with k1:
#     st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
# with k2:
#     st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
# with k3:
#     st.metric("Minimum Rides", f"{max(filtered_predictions['predicted_demand'].min(), 0):.0f}")

# progress_bar.progress(4 / N_STEPS)

# # ---------- Table ----------
# st.subheader("Top 10 Zones")
# st.dataframe(
#     filtered_predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)
#     .reset_index(drop=True),
#     use_container_width=True,
# )

# # ---------- Optional: per-zone history charts ----------
# top10_zones = (
#     filtered_predictions.sort_values("predicted_demand", ascending=False)
#     .head(10)["zone"]
#     .dropna()
#     .tolist()
# )
# for zone in top10_zones:
#     fig = plot_prediction(
#         features=features[features["zone"] == zone],
#         prediction=filtered_predictions[filtered_predictions["zone"] == zone],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# frontend/app.py

# import sys
# from pathlib import Path

# # Make src importable
# parent_dir = str(Path(__file__).parent.parent)
# sys.path.append(parent_dir)

# # Avoid old cached version on Streamlit Cloud
# from importlib import reload as _reload
# import src.inference as _inf
# _inf = _reload(_inf)

# import zipfile
# import folium
# import geopandas as gpd
# import pandas as pd
# import requests
# import streamlit as st
# from branca.colormap import LinearColormap
# from streamlit_folium import st_folium

# from src.config import DATA_DIR
# from src.plot_utils import plot_prediction
# from src.inference import fetch_next_hour_predictions  # next-hour fetch


# # ---------------- Page config + subtle UI polish ----------------
# st.set_page_config(
#     page_title="NYC Taxi Demand — Next Hour",
#     page_icon="🚕",
#     layout="wide",
# )

# # Minimal CSS polish
# st.markdown(
#     """
#     <style>
#       /* Page background + fonts */
#       .main { padding-top: 0rem; }
#       .block-container { padding-top: 1rem; }
#       h1, h2, h3, h4 { letter-spacing: .2px; }

#       /* Hero card */
#       .hero {
#         border-radius: 20px;
#         padding: 24px 28px;
#         background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
#         color: white;
#         box-shadow: 0 10px 25px rgba(2, 6, 23, .25);
#       }
#       .hero h2 { margin: 0 0 4px 0; font-weight: 700; }
#       .hero p { margin: 0; opacity: .95; }

#       /* Pills */
#       .pill {
#         display: inline-block;
#         padding: 4px 10px;
#         border-radius: 999px;
#         background: rgba(255,255,255,.15);
#         border: 1px solid rgba(255,255,255,.2);
#         color: #fff;
#         font-size: 12px;
#         margin-right: 8px;
#       }
#       /* Cards */
#       .card {
#         border-radius: 16px;
#         border: 1px solid rgba(2, 6, 23, .08);
#         background: #fff;
#         padding: 18px;
#       }

#       /* Center images nicer in hero-right */
#       .stImage > img { border-radius: 12px; }

#       /* Tighten KPI metrics spacing */
#       div[data-testid="metric-container"] {
#         background: #fff;
#         border: 1px solid rgba(2, 6, 23, .08);
#         border-radius: 16px;
#         padding: 12px 16px;
#         box-shadow: 0 4px 18px rgba(2, 6, 23, .06);
#       }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ---------------- Images (royalty-free) ----------------
# NYC_TAXI_URL = (
#     "https://images.unsplash.com/photo-1483721168571-c0895f4432c7"
#     "?auto=format&fit=crop&q=60&w=1100"
# )
# NYC_SKYLINE_URL = (
#     "https://images.unsplash.com/photo-1534430480872-3498386e7856"
#     "?auto=format&fit=crop&q=60&w=1100"
# )

# # ---------------- Hero header ----------------
# c1, c2 = st.columns([1.25, 1], gap="large")
# with c1:
#     st.markdown('<div class="hero">', unsafe_allow_html=True)
#     st.markdown("### 🚕 NYC Taxi Demand", unsafe_allow_html=True)
#     st.markdown(
#         "<p>Predicting next-hour rides per taxi zone. "
#         "If the newest batch isn’t ready yet, we’ll show the latest available hour.</p>",
#         unsafe_allow_html=True,
#     )
#     now_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S")
#     st.markdown(
#         f'<span class="pill">NY Time: {now_ny}</span>'
#         f'<span class="pill">Live</span>',
#         unsafe_allow_html=True,
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

# with c2:
#     st.image(NYC_TAXI_URL, caption="Manhattan yellow taxis", use_container_width=True)

# st.divider()


# # ---------------- Session/state ----------------
# if "map_created" not in st.session_state:
#     st.session_state.map_created = False


# # ---------------- Data helpers ----------------
# @st.cache_data(show_spinner=False)
# def load_shape_data_file(
#     data_dir: Path, url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
# ) -> gpd.GeoDataFrame:
#     """Download once and cache NYC taxi zones shapefile."""
#     data_dir = Path(data_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     zip_path = data_dir / "taxi_zones.zip"
#     extract_path = data_dir / "taxi_zones"
#     shapefile_path = extract_path / "taxi_zones.shp"

#     if not zip_path.exists():
#         response = requests.get(url, timeout=20)
#         response.raise_for_status()
#         zip_path.write_bytes(response.content)

#     if not shapefile_path.exists():
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(extract_path)

#     gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
#     return gdf


# @st.cache_data(show_spinner=False)
# def load_lookup_df() -> pd.DataFrame:
#     """Load taxi zone lookup once and cache; add safe 'zone' column."""
#     lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
#     df = pd.read_csv(lookup_file)
#     if "zone" not in df.columns and "Zone" in df.columns:
#         df = df.rename(columns={"Zone": "zone"})
#     return df


# def _attach_zone_names(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
#     if df is None or df.empty:
#         return df
#     df = df.copy()
#     df["predicted_demand"] = pd.to_numeric(df["predicted_demand"], errors="coerce").clip(lower=0)
#     if "pickup_hour" in df.columns:
#         # keep NY time for UI
#         df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     df = df.merge(
#         lookup_df[["LocationID", "zone", "Borough"]] if "Borough" in lookup_df.columns else lookup_df[["LocationID", "zone"]],
#         left_on="pickup_location_id",
#         right_on="LocationID",
#         how="left",
#     ).drop(columns=["LocationID"])
#     return df


# def create_taxi_map(shapefile_path: Path, prediction_data: pd.DataFrame):
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
#         vmin=float(nyc_zones["predicted_demand"].min()) if len(nyc_zones) else 0.0,
#         vmax=float(nyc_zones["predicted_demand"].max()) if len(nyc_zones) else 1.0,
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


# def fetch_predictions_with_fallback() -> tuple[pd.DataFrame, bool, pd.Timestamp]:
#     """
#     Try 'next hour' first. If empty, return latest available hour within last 6 hours.
#     Returns: (predictions_df, used_fallback, shown_timestamp_utc)
#     """
#     # Expect next-hour in UTC
#     now_utc = pd.Timestamp.now(tz="UTC")
#     expected_next_hour = (now_utc + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

#     preds = fetch_next_hour_predictions()  # returns UTC pickup_hour in FG
#     if not preds.empty:
#         return preds.copy(), False, expected_next_hour

#     # fallback: latest available within last 6 hours
#     history = _inf.fetch_predictions(hours=6)
#     if history.empty:
#         return history, True, expected_next_hour

#     latest_ts = pd.to_datetime(history["pickup_hour"], utc=True).max()
#     return history[history["pickup_hour"] == latest_ts].copy(), True, latest_ts


# # ---------------- Sidebar flow + progress ----------------
# with st.sidebar:
#     st.image(NYC_SKYLINE_URL, use_container_width=True)
#     st.markdown("### Controls")
#     refresh_clicked = st.button("🔄 Refresh now")

# progress_bar = st.sidebar.progress(0)
# N_STEPS = 4

# with st.spinner("Downloading NYC taxi zones…"):
#     _ = load_shape_data_file(DATA_DIR)
#     progress_bar.progress(1 / N_STEPS)

# with st.spinner("Fetching features from store (FV + FG fallback)…"):
#     # robust loader from src.inference (auto-creates FV, falls back to FG)
#     current_utc = pd.Timestamp.now(tz="Etc/UTC")
#     try:
#         features = _inf.load_batch_of_features_from_store(current_utc)
#     except Exception as e:
#         st.error(
#             "Could not fetch features from Hopsworks (Feature View + FG fallback).\n\n"
#             "Common causes: FV not created yet, permissions, or empty time window.\n\n"
#             f"Details: {type(e).__name__}: {e}"
#         )
#         st.stop()
#     progress_bar.progress(2 / N_STEPS)

# # ---------------- Predictions (explicit next-hour check) ----------------
# with st.spinner("Fetching predictions…"):
#     predictions_raw, used_fallback, shown_ts_utc = fetch_predictions_with_fallback()

#     if predictions_raw.empty:
#         st.error("No predictions found (including fallback search in last 6 hours).")
#         st.stop()

#     # Load lookup + attach zone/borough
#     lookup_df = load_lookup_df()
#     predictions = _attach_zone_names(predictions_raw, lookup_df)

#     # Also attach zones to features and keep pickup_hour in NY time for charts
#     if "pickup_hour" in features.columns:
#         features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
#     features = features.merge(
#         lookup_df[["LocationID", "zone", "Borough"]] if "Borough" in lookup_df.columns else lookup_df[["LocationID", "zone"]],
#         left_on="pickup_location_id",
#         right_on="LocationID",
#         how="left",
#     ).drop(columns=["LocationID"])

#     progress_bar.progress(3 / N_STEPS)

# # Status line under hero
# status_col1, status_col2, status_col3 = st.columns([1.1, 1, 1.2])
# with status_col1:
#     if used_fallback:
#         st.info(
#             f"⏳ Next-hour not ready — showing **latest available** hour: "
#             f"{shown_ts_utc.tz_convert('America/New_York'):%Y-%m-%d %H:%M} (NY).",
#             icon="ℹ️",
#         )
#     else:
#         st.success(
#             f"✅ Showing **next-hour** predictions for "
#             f"{shown_ts_utc.tz_convert('America/New_York'):%Y-%m-%d %H:%M} (NY)."
#         )
# with status_col2:
#     if st.button("Try next hour again"):
#         st.rerun()
# with status_col3:
#     st.download_button(
#         "⬇️ Download predictions (CSV)",
#         data=predictions.to_csv(index=False).encode("utf-8"),
#         file_name="nyc_next_hour_predictions.csv",
#         mime="text/csv",
#     )

# st.divider()

# # ---------------- Filters ----------------
# lookup_df = load_lookup_df()
# boroughs = (
#     ["All Boroughs"]
#     + sorted(lookup_df["Borough"].dropna().unique().tolist())
#     if "Borough" in lookup_df.columns
#     else ["All Boroughs"]
# )
# zones = ["All Zones"] + sorted(predictions["zone"].dropna().unique().tolist())

# f1, f2, f3 = st.columns([1, 1, 2])
# with f1:
#     selected_borough = st.selectbox("Borough", boroughs, index=0)
# with f2:
#     selected_zone = st.selectbox("Zone", zones, index=0)
# with f3:
#     st.caption(
#         "Tip: pick a borough first to narrow the zone list. "
#         "Use the refresh buttons above if your inference pipeline just ran."
#     )

# # Apply filters
# filtered = predictions.copy()
# if selected_borough != "All Boroughs" and "Borough" in filtered.columns:
#     filtered = filtered[filtered["Borough"] == selected_borough]
# if selected_zone != "All Zones":
#     filtered = filtered[filtered["zone"] == selected_zone]

# st.caption(
#     "Mode: **next hour**" if not used_fallback else "Mode: **latest available hour** (next hour not ready yet)"
# )

# # ---------------- Tabs: Map | Table | Trends | About ----------------
# tab_map, tab_table, tab_trends, tab_about = st.tabs(["🗺️ Map", "📋 Table", "📈 Trends", "ℹ️ About"])

# # Map tab
# with tab_map:
#     shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"
#     with st.spinner("Rendering map…"):
#         st.subheader("Choropleth by Predicted Demand")
#         create_taxi_map(shapefile_path, filtered)
#         if st.session_state.map_created:
#             st_folium(st.session_state.map_obj, width=1200, height=600, returned_objects=[])

#     k1, k2, k3 = st.columns(3)
#     with k1:
#         st.metric("Average Rides", f"{filtered['predicted_demand'].mean():.0f}")
#     with k2:
#         st.metric("Maximum Rides", f"{filtered['predicted_demand'].max():.0f}")
#     with k3:
#         st.metric("Minimum Rides", f"{max(filtered['predicted_demand'].min(), 0):.0f}")

# # Table tab
# with tab_table:
#     st.subheader("Top 15 Zones")
#     st.dataframe(
#         filtered.sort_values("predicted_demand", ascending=False)
#         .head(15)
#         .reset_index(drop=True),
#         use_container_width=True,
#     )
#     st.download_button(
#         "⬇️ Download filtered table (CSV)",
#         data=filtered.to_csv(index=False).encode("utf-8"),
#         file_name="nyc_filtered_predictions.csv",
#         mime="text/csv",
#     )

# # Trends tab
# with tab_trends:
#     st.subheader("Per-Zone Trends")
#     # Take top zones in the current filtered set
#     top_zones = (
#         filtered.sort_values("predicted_demand", ascending=False)
#         .head(10)["zone"]
#         .dropna()
#         .tolist()
#     )
#     if not top_zones:
#         st.info("Pick a borough/zone or expand filters to see trend charts.")
#     for zone in top_zones:
#         fig = plot_prediction(
#             features=features[features["zone"] == zone],
#             prediction=filtered[filtered["zone"] == zone],
#         )
#         st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# # About tab
# with tab_about:
#     st.subheader("About this app")
#     st.markdown(
#         """
#         **What you’re seeing**

#         - **Next-hour predictions** per NYC Taxi Zone.
#         - If the newest batch isn’t ready yet, we automatically show the **latest available hour** instead.
#         - You can filter by **Borough** and **Zone**, view the **map**, **table**, and **per-zone trends**.

#         **Data + Infra**

#         - Feature pipeline writes hourly aggregates to a Hopsworks **Feature Group**.
#         - Feature View (FV) is auto-ensured in code; if FV fetch fails, we **fallback** to FG.
#         - Inference pipeline writes predictions to a **model_predictions** Feature Group.

#         **Tips**

#         - Use **Refresh** or **Try next hour again** after your GitHub Actions inference job finishes.
#         - Download CSVs for the current view from the buttons above.
#         """
#     )

# progress_bar.progress(4 / N_STEPS)

# # Manual refresh at end if requested
# if refresh_clicked:
#     st.rerun()


# frontend/app.py

import sys
from pathlib import Path
from typing import Tuple

# Make src importable
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Always use fresh src.inference (avoid stale import on Streamlit Cloud)
from importlib import reload as _reload
import src.inference as _inf
_inf = _reload(_inf)

import zipfile
import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.plot_utils import plot_prediction
from src.inference import fetch_next_hour_predictions  # next-hour fetch

# ---------------- Page config ----------------
st.set_page_config(
    page_title="NYC Taxi Demand — Next Hour",
    page_icon="🚕",
    layout="wide",
)

# ---------------- Light CSS polish ----------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; }
      .hero {
        border-radius: 20px;
        padding: 22px 26px;
        background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
        color: white;
        box-shadow: 0 10px 28px rgba(2, 6, 23, .25);
      }
      .hero h2 { margin: 0 0 4px 0; font-weight: 750; letter-spacing: .2px; }
      .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.2);
        color:#fff; font-size:12px; margin-right:8px;
      }
      .card {
        border-radius: 16px; border: 1px solid rgba(2,6,23,.08);
        background: #fff; padding: 18px;
        box-shadow: 0 6px 18px rgba(2, 6, 23, .07);
      }
      div[data-testid="metric-container"] {
        background: #fff; border: 1px solid rgba(2,6,23,.08);
        border-radius: 14px; padding: 10px 14px;
        box-shadow: 0 4px 18px rgba(2, 6, 23, .06);
      }
      /* Make tab headers a bit bolder */
      button[role="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Images ----------------
NYC_TAXI_URL = (
    "https://images.unsplash.com/photo-1483721168571-c0895f4432c7?auto=format&fit=crop&q=60&w=1100"
)
NYC_SKYLINE_URL = (
    "https://images.unsplash.com/photo-1534430480872-3498386e7856?auto=format&fit=crop&q=60&w=1100"
)

# ---------------- Hero ----------------
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("### 🚕 NYC Taxi Demand", unsafe_allow_html=True)
    st.markdown(
        "<p>Predicting next-hour rides per taxi zone. "
        "If the newest batch isn’t ready yet, we show the latest available hour.</p>",
        unsafe_allow_html=True,
    )
    now_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f'<span class="pill">NY Time: {now_ny}</span>'
        f'<span class="pill">Live</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.image(NYC_TAXI_URL, caption="Manhattan yellow taxis", use_container_width=True)

st.divider()

# ---------------- Session / state ----------------
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# ---------------- Data helpers ----------------
@st.cache_data(show_spinner=False)
def load_shape_data_file(
    data_dir: Path, url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
) -> gpd.GeoDataFrame:
    """Download once and cache NYC taxi zones shapefile."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    if not shapefile_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
    return gdf


@st.cache_data(show_spinner=False)
def load_lookup_df() -> pd.DataFrame:
    """Load taxi zone lookup once and cache; ensure 'zone' column exists."""
    lookup_file = Path(__file__).parent.parent / "taxi_zone_lookup.csv"
    df = pd.read_csv(lookup_file)
    if "zone" not in df.columns and "Zone" in df.columns:
        df = df.rename(columns={"Zone": "zone"})
    return df


def _attach_zone_names(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["predicted_demand"] = pd.to_numeric(out["predicted_demand"], errors="coerce").clip(lower=0)
    if "pickup_hour" in out.columns:
        out["pickup_hour"] = pd.to_datetime(out["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
    out = out.merge(
        lookup_df[["LocationID", "zone", "Borough"]] if "Borough" in lookup_df.columns else lookup_df[["LocationID", "zone"]],
        left_on="pickup_location_id",
        right_on="LocationID",
        how="left",
    ).drop(columns=["LocationID"])
    return out


def create_taxi_map(shapefile_path: Path, prediction_data: pd.DataFrame):
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
        vmin=float(nyc_zones["predicted_demand"].min()) if len(nyc_zones) else 0.0,
        vmax=float(nyc_zones["predicted_demand"].max()) if len(nyc_zones) else 1.0,
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


def fetch_predictions_with_fallback() -> Tuple[pd.DataFrame, bool, pd.Timestamp]:
    """
    Try 'next hour' first. If empty, return latest available hour within last 6 hours.
    Returns: (predictions_df, used_fallback, shown_timestamp_utc)
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    expected_next_hour = (now_utc + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    preds = fetch_next_hour_predictions()  # UTC pickup_hour
    if not preds.empty:
        return preds.copy(), False, expected_next_hour

    history = _inf.fetch_predictions(hours=6)
    if history.empty:
        return history, True, expected_next_hour

    latest_ts = pd.to_datetime(history["pickup_hour"], utc=True).max()
    return history[history["pickup_hour"] == latest_ts].copy(), True, latest_ts


# ---------------- Sidebar ----------------
with st.sidebar:
    st.image(NYC_SKYLINE_URL, use_container_width=True)
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=False)
    if st.button("🔄 Refresh now"):
        st.rerun()  # <-- fixed: use st.rerun()

progress_bar = st.sidebar.progress(0)
N_STEPS = 4

# ---------------- Load static data ----------------
with st.spinner("Downloading NYC taxi zones…"):
    _ = load_shape_data_file(DATA_DIR)
    progress_bar.progress(1 / N_STEPS)

# ---------------- Load features (FV with FG fallback) ----------------
with st.spinner("Fetching features from store…"):
    current_utc = pd.Timestamp.now(tz="Etc/UTC")
    try:
        features = _inf.load_batch_of_features_from_store(current_utc)
    except Exception as e:
        st.error(
            "Could not fetch features from Hopsworks (Feature View + FG fallback).\n\n"
            "Common causes: FV not created yet, permissions, or empty time window.\n\n"
            f"Details: {type(e).__name__}: {e}"
        )
        st.stop()
    progress_bar.progress(2 / N_STEPS)

# ---------------- Predictions ----------------
with st.spinner("Fetching predictions…"):
    predictions_raw, used_fallback, shown_ts_utc = fetch_predictions_with_fallback()
    if predictions_raw.empty:
        st.error("No predictions found (even after fallback search over last 6 hours).")
        st.stop()

    lookup_df = load_lookup_df()
    predictions = _attach_zone_names(predictions_raw, lookup_df)

    if "pickup_hour" in features.columns:
        features["pickup_hour"] = pd.to_datetime(features["pickup_hour"], utc=True).dt.tz_convert("America/New_York")
    features = features.merge(
        lookup_df[["LocationID", "zone", "Borough"]] if "Borough" in lookup_df.columns else lookup_df[["LocationID", "zone"]],
        left_on="pickup_location_id",
        right_on="LocationID",
        how="left",
    ).drop(columns=["LocationID"])

    progress_bar.progress(3 / N_STEPS)

# ---------------- Status line ----------------
s1, s2, s3 = st.columns([1.1, 1, 1.2])
with s1:
    if used_fallback:
        st.info(
            f"⏳ Next-hour not ready — showing latest available hour: "
            f"{shown_ts_utc.tz_convert('America/New_York'):%Y-%m-%d %H:%M} (NY).",
            icon="ℹ️",
        )
    else:
        st.success(
            f"✅ Showing next-hour predictions for "
            f"{shown_ts_utc.tz_convert('America/New_York'):%Y-%m-%d %H:%M} (NY)."
        )
with s2:
    if st.button("Try next hour again"):
        st.rerun()  # <-- fixed: use st.rerun()
with s3:
    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="nyc_next_hour_predictions.csv",
        mime="text/csv",
    )

st.divider()

# ---------------- Filters ----------------
boroughs = (
    ["All Boroughs"]
    + sorted(predictions["Borough"].dropna().unique().tolist())
    if "Borough" in predictions.columns
    else ["All Boroughs"]
)
zones = ["All Zones"] + sorted(predictions["zone"].dropna().unique().tolist())

f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    selected_borough = st.selectbox("Borough", boroughs, index=0)
with f2:
    selected_zone = st.selectbox("Zone", zones, index=0)
with f3:
    st.caption("Tip: pick a borough to narrow the zone list. Use refresh if your pipeline just ran.")

# Apply filters
filtered = predictions.copy()
if selected_borough != "All Boroughs" and "Borough" in filtered.columns:
    filtered = filtered[filtered["Borough"] == selected_borough]
if selected_zone != "All Zones":
    filtered = filtered[filtered["zone"] == selected_zone]

st.caption(
    "Mode: **next hour**" if not used_fallback else "Mode: **latest available hour** (next hour not ready yet)"
)

# ---------------- Tabs ----------------
tab_map, tab_table, tab_trends, tab_about = st.tabs(["🗺️ Map", "📋 Table", "📈 Trends", "ℹ️ About"])

with tab_map:
    shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"
    with st.spinner("Rendering map…"):
        st.subheader("Choropleth by Predicted Demand")
        create_taxi_map(shapefile_path, filtered)
        if st.session_state.map_created:
            st_folium(st.session_state.map_obj, width=1200, height=600, returned_objects=[])

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Average Rides", f"{filtered['predicted_demand'].mean():.0f}")
    with k2:
        st.metric("Maximum Rides", f"{filtered['predicted_demand'].max():.0f}")
    with k3:
        st.metric("Minimum Rides", f"{max(filtered['predicted_demand'].min(), 0):.0f}")

with tab_table:
    st.subheader("Top 15 Zones")
    st.dataframe(
        filtered.sort_values("predicted_demand", ascending=False)
        .head(15)
        .reset_index(drop=True),
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download filtered table (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="nyc_filtered_predictions.csv",
        mime="text/csv",
    )

with tab_trends:
    st.subheader("Per-Zone Trends")
    top_zones = (
        filtered.sort_values("predicted_demand", ascending=False)
        .head(10)["zone"]
        .dropna()
        .tolist()
    )
    if not top_zones:
        st.info("Pick a borough/zone or expand filters to see trend charts.")
    for zone in top_zones:
        fig = plot_prediction(
            features=features[features["zone"] == zone],
            prediction=filtered[filtered["zone"] == zone],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
        **What you’re seeing**
        - Next-hour predictions per NYC Taxi Zone; if not ready, we show the latest available hour.
        - Filter by Borough and Zone, view a choropleth map, table, and per-zone trend charts.

        **Data & Infra**
        - Feature pipeline writes hourly aggregates to a Hopsworks Feature Group.
        - Feature View is auto-ensured in backend; if FV fetch fails, we fallback to FG.
        - Inference pipeline writes predictions to a model_predictions Feature Group.

        **Tips**
        - Use Refresh / Try next hour again after pipelines run.
        - Download CSVs for the current view.
        """
    )

progress_bar.progress(4 / N_STEPS)

# ---------------- Optional auto-refresh ----------------
if auto_refresh:
    # Re-run once a minute without user interaction
    st.caption("Auto-refresh is ON (every 60s).")
    st.experimental_singleton.clear() if hasattr(st, "experimental_singleton") else None  # harmless
    st.cache_data.clear()  # ensure we refetch on rerun
    st.experimental_set_query_params(ref=pd.Timestamp.now().isoformat())
    st.timeout = 60  # no-op var to show intent
    st.rerun()


# Plotting helper is optional; keep if you already have src.plot_utils.plot_prediction
# from src.plot_utils import plot_prediction
# for zone in top10_zones:
#     fig = plot_prediction(
#         features=features[features["zone"] == zone],
#         prediction=filtered_predictions[filtered_predictions["zone"] == zone],
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)
