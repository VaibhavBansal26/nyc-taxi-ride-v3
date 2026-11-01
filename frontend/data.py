"""Cached data access for the Streamlit app."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import src.config as config
from src.data_source import DemoSource, HopsworksSource, resolve_source
from src.data_utils import load_zone_lookup

PREDICTION_TTL = 300  # seconds


def requested_mode() -> str:
    return st.session_state.get("app_mode", config.APP_MODE)


@st.cache_resource(show_spinner="Connecting to data source…")
def _source_for(mode: str) -> tuple[HopsworksSource | DemoSource, str | None]:
    return resolve_source(mode)


def get_source() -> tuple[HopsworksSource | DemoSource, str | None]:
    try:
        return _source_for(requested_mode())
    except Exception as exc:
        st.error(
            f"Could not initialise the **{requested_mode()}** data source: `{type(exc).__name__}: {str(exc)[:300]}`\n\n"
            "Check `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT_NAME`, or pick **Demo** in the sidebar."
        )
        st.stop()


def target_hour() -> pd.Timestamp:
    """The hour being predicted: ``now`` rounded up (UTC)."""
    return pd.Timestamp.now(tz="UTC").ceil("h")


@st.cache_data(ttl=PREDICTION_TTL, show_spinner="Fetching predictions…")
def next_hour_predictions(mode: str, target_iso: str, _source) -> pd.DataFrame:
    return _source.next_hour_predictions(pd.Timestamp(target_iso))


@st.cache_data(ttl=PREDICTION_TTL, show_spinner="Fetching recent predictions…")
def recent_predictions(mode: str, hours: int, _source) -> pd.DataFrame:
    return _source.recent_predictions(hours)


@st.cache_data(ttl=PREDICTION_TTL, show_spinner="Fetching observed rides…")
def recent_actuals(mode: str, hours: int, _source) -> pd.DataFrame:
    return _source.recent_actuals(hours)


@st.cache_data(ttl=PREDICTION_TTL, show_spinner="Building feature windows…")
def latest_features(mode: str, target_iso: str, _source) -> pd.DataFrame:
    return _source.latest_features(pd.Timestamp(target_iso))


def clear_prediction_caches() -> None:
    for fn in (next_hour_predictions, recent_predictions, recent_actuals, latest_features):
        fn.clear()


@st.cache_data(show_spinner=False)
def zone_lookup() -> pd.DataFrame:
    return load_zone_lookup()


def attach_zone_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.merge(zone_lookup(), on="pickup_location_id", how="left")
    out["zone"] = out["zone"].fillna("Unknown")
    out["borough"] = out["borough"].fillna("Unknown")
    if "pickup_hour" in out.columns:
        out["pickup_hour_ny"] = pd.to_datetime(out["pickup_hour"], utc=True).dt.tz_convert(config.LOCAL_TIMEZONE)
    return out


def _find_shapefile(folder: Path) -> Path | None:
    return next(iter(sorted(folder.rglob("*.shp"))), None)


@st.cache_data(show_spinner="Downloading NYC taxi zone shapes…")
def zones_geojson() -> dict:
    """GeoJSON of taxi zones (simplified) with LocationID / zone / borough properties."""
    import geopandas as gpd

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"

    shapefile = _find_shapefile(extract_path) if extract_path.exists() else None
    if shapefile is None:
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            resp = requests.get(config.TAXI_ZONES_URL, timeout=60)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_path)
        shapefile = _find_shapefile(extract_path)
    if shapefile is None:
        raise FileNotFoundError("taxi_zones.shp not found after extracting the TLC archive")

    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    gdf = gdf[["LocationID", "zone", "borough", "geometry"]].copy()
    gdf["geometry"] = gdf["geometry"].simplify(0.0004, preserve_topology=True)
    gdf["LocationID"] = gdf["LocationID"].astype(int)
    return json.loads(gdf.to_json())
