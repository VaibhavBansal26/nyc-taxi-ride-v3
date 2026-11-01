"""Main page: next-hour predictions per zone."""

from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

import src.config as config
from frontend import data, ui
from src.plot_utils import plot_borough_share, plot_prediction, plot_top_zones

FALLBACK_HOURS = 6


def _load_predictions(source, mode: str) -> tuple[pd.DataFrame, pd.Timestamp, bool]:
    """Next-hour predictions, else the latest hour available in the last ``FALLBACK_HOURS``."""
    target = data.target_hour()
    preds = data.next_hour_predictions(mode, target.isoformat(), source)
    if not preds.empty:
        return preds, target, False
    history = data.recent_predictions(mode, FALLBACK_HOURS, source)
    if history.empty:
        return history, target, True
    latest = pd.to_datetime(history["pickup_hour"], utc=True).max()
    return history[pd.to_datetime(history["pickup_hour"], utc=True) == latest].copy(), latest, True


def _build_map(geojson: dict, preds: pd.DataFrame) -> folium.Map:
    demand = dict(zip(preds["pickup_location_id"].astype(int), preds["predicted_demand"].astype(float), strict=False))
    for feature in geojson["features"]:
        props = feature["properties"]
        props["predicted_demand"] = float(demand.get(int(props["LocationID"]), 0.0))

    values = [f["properties"]["predicted_demand"] for f in geojson["features"]]
    vmax = max(values) if values else 1.0
    colormap = LinearColormap(
        colors=["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
        vmin=0.0,
        vmax=max(vmax, 1.0),
        caption="Predicted rides next hour",
    )
    m = folium.Map(location=[40.73, -73.93], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    colormap.add_to(m)
    folium.GeoJson(
        geojson,
        style_function=lambda f: {
            "fillColor": colormap(f["properties"]["predicted_demand"]),
            "color": "#334155",
            "weight": 0.6,
            "fillOpacity": 0.75,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#0f172a", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "borough", "predicted_demand"],
            aliases=["Zone", "Borough", "Predicted rides"],
            localize=True,
            sticky=False,
            style="background:#fff;color:#0f172a;font-family:sans-serif;font-size:12px;padding:8px;border-radius:8px;",
        ),
    ).add_to(m)
    return m


def render() -> None:
    source, note = data.get_source()
    mode = source.mode
    status = source.status()

    preds_raw, shown_hour, used_fallback = _load_predictions(source, mode)
    shown_ny = shown_hour.tz_convert(config.LOCAL_TIMEZONE)

    ui.hero(
        "NYC Taxi Demand - next hour",
        "Predicted yellow-taxi pickups per TLC zone for the coming hour, from a LightGBM model fed with the last 28 days of hourly rides.",
        [
            ui.mode_pill(mode),
            (f"Target hour: {shown_ny:%a %d %b, %H:%M} NY", ""),
            (f"NY time: {ui.ny_now():%H:%M}", ""),
        ],
    )
    if note:
        st.warning(note, icon="⚠️")
    if used_fallback and not preds_raw.empty:
        st.info(
            f"Next-hour predictions are not in the store yet - showing the latest available hour ({shown_ny:%H:%M} NY).",
            icon="⏳",
        )
    if preds_raw.empty:
        st.error(
            f"No predictions found for the last {FALLBACK_HOURS} hours. "
            "Run the inference pipeline (`python -m pipelines.inference_pipeline`) or switch to demo mode in the sidebar."
        )
        st.stop()

    preds = data.attach_zone_names(preds_raw)

    # ---------------------------------------------------------------- filters
    f1, f2, f3 = st.columns([1.2, 1.6, 1])
    boroughs = sorted(preds["borough"].unique().tolist())
    with f1:
        selected_boroughs = st.multiselect("Borough", boroughs, default=[], placeholder="All boroughs")
    pool = preds[preds["borough"].isin(selected_boroughs)] if selected_boroughs else preds
    with f2:
        selected_zones = st.multiselect(
            "Zone", sorted(pool["zone"].unique().tolist()), default=[], placeholder="All zones"
        )
    with f3:
        min_demand = st.slider("Min predicted rides", 0, int(max(preds["predicted_demand"].max(), 1)), 0)

    filtered = pool.copy()
    if selected_zones:
        filtered = filtered[filtered["zone"].isin(selected_zones)]
    filtered = filtered[filtered["predicted_demand"] >= min_demand].sort_values("predicted_demand", ascending=False)

    # ---------------------------------------------------------------- KPIs
    total = filtered["predicted_demand"].sum()
    busiest = filtered.iloc[0] if len(filtered) else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted rides", f"{total:,.0f}", help="Sum of the predicted pickups across the selected zones")
    k2.metric(
        "Busiest zone",
        f"{busiest['predicted_demand']:.0f} rides" if busiest is not None else "-",
        busiest["zone"] if busiest is not None else None,
        delta_color="off",
    )
    k3.metric("Zones shown", f"{len(filtered)}", f"of {len(preds)}", delta_color="off")
    k4.metric("Mean per zone", f"{filtered['predicted_demand'].mean():.1f}" if len(filtered) else "-")

    st.caption(f"Source: {status.detail}. Figures are rounded ride counts; hours are shown in New York time.")

    tab_map, tab_top, tab_table, tab_trends = st.tabs(["🗺️ Map", "🏆 Top zones", "📋 Table", "📈 Zone history"])

    with tab_map:
        try:
            geojson = data.zones_geojson()
            st_folium(_build_map(geojson, filtered), height=620, width="stretch", returned_objects=[], key="demand_map")
        except Exception as exc:  # network / GDAL problems should not take the page down
            st.error(f"Could not render the zone map: {type(exc).__name__}: {exc}")
            st.dataframe(filtered[["zone", "borough", "predicted_demand"]], width="stretch", hide_index=True)

    with tab_top:
        if filtered.empty:
            st.info("No zones match the current filters.")
        else:
            c1, c2 = st.columns([1.6, 1])
            with c1:
                st.plotly_chart(plot_top_zones(filtered, n=15), width="stretch", key="top_zones")
            with c2:
                st.markdown("**Share by borough**")
                st.plotly_chart(plot_borough_share(filtered), width="stretch", key="borough_share")

    with tab_table:
        table = filtered[["pickup_location_id", "zone", "borough", "predicted_demand"]].rename(
            columns={
                "pickup_location_id": "Zone ID",
                "zone": "Zone",
                "borough": "Borough",
                "predicted_demand": "Predicted rides",
            }
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Predicted rides": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=float(max(preds["predicted_demand"].max(), 1))
                )
            },
        )
        st.download_button(
            "⬇️ Download as CSV",
            data=preds.drop(columns=["pickup_hour_ny"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name=f"nyc_taxi_predictions_{shown_hour:%Y%m%dT%H%M}Z.csv",
            mime="text/csv",
        )

    with tab_trends:
        st.markdown("Last 7 days of observed pickups for a zone, with the model's next-hour prediction.")
        zone_options = filtered["zone"].tolist() if not filtered.empty else preds["zone"].tolist()
        chosen = st.selectbox("Zone", zone_options, index=0, key="trend_zone")
        try:
            features = data.latest_features(mode, data.target_hour().isoformat(), source)
            row = preds[preds["zone"] == chosen].iloc[0]
            feat_row = features[features["pickup_location_id"] == row["pickup_location_id"]]
            if feat_row.empty:
                st.info("No feature window available for this zone yet.")
            else:
                st.plotly_chart(
                    plot_prediction(feat_row, row["predicted_demand"], title=f"{chosen} ({row['borough']})"),
                    width="stretch",
                    key="trend_chart",
                )
        except Exception as exc:
            st.error(f"Could not load feature history: {type(exc).__name__}: {exc}")
