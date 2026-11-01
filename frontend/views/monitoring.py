"""Model monitoring: predictions vs observed rides over the recent past."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import src.config as config
from frontend import data, ui
from src.plot_utils import plot_actual_vs_predicted, plot_mae_by_hour


def _merge(actuals: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    if actuals.empty or preds.empty:
        return pd.DataFrame(columns=["pickup_location_id", "pickup_hour", "rides", "predicted_demand"])
    a = actuals.copy()
    p = preds.copy()
    a["pickup_hour"] = pd.to_datetime(a["pickup_hour"], utc=True)
    p["pickup_hour"] = pd.to_datetime(p["pickup_hour"], utc=True)
    merged = a.merge(p, on=["pickup_location_id", "pickup_hour"], how="inner")
    merged["abs_error"] = (merged["predicted_demand"] - merged["rides"]).abs()
    return merged


def render() -> None:
    source, note = data.get_source()
    mode = source.mode

    ui.hero(
        "Model monitoring",
        "How close were the hourly predictions to what actually happened? Errors are computed where both a prediction and the observed count exist.",
        [ui.mode_pill(mode), (f"NY time: {ui.ny_now():%H:%M}", "")],
    )
    if note:
        st.warning(note, icon="⚠️")

    hours = st.slider("Look-back window (hours)", min_value=12, max_value=24 * 7, value=48, step=12)
    actuals = data.recent_actuals(mode, hours, source)
    preds = data.recent_predictions(mode, hours, source)
    merged = _merge(actuals, preds)

    if merged.empty:
        st.error(
            "No overlapping predictions and observations in this window. "
            "In live mode both the feature pipeline and the inference pipeline must have run for these hours."
        )
        st.stop()

    merged = data.attach_zone_names(merged)
    mae = merged["abs_error"].mean()
    total_actual = merged["rides"].sum()
    total_pred = merged["predicted_demand"].sum()
    bias = (merged["predicted_demand"] - merged["rides"]).mean()
    hours_covered = merged["pickup_hour"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (rides)", f"{mae:.2f}", help="Mean absolute error per zone and hour")
    k2.metric("Bias", f"{bias:+.2f}", help="Positive = over-prediction on average")
    k3.metric(
        "Total predicted",
        f"{total_pred:,.0f}",
        f"{(total_pred - total_actual) / max(total_actual, 1):+.1%} vs {total_actual:,.0f} actual",
        delta_color="off",
    )
    k4.metric("Hours", f"{hours_covered}", f"{merged['pickup_location_id'].nunique()} zones", delta_color="off")

    tab_hourly, tab_zones, tab_zone_detail = st.tabs(["⏱️ MAE by hour", "🏙️ Hardest zones", "🔍 Zone detail"])

    with tab_hourly:
        by_hour = (
            merged.groupby("pickup_hour_ny", as_index=False)["abs_error"]
            .mean()
            .rename(columns={"pickup_hour_ny": "pickup_hour", "abs_error": "mae"})
        )
        st.plotly_chart(plot_mae_by_hour(by_hour), width="stretch", key="mae_by_hour")

    with tab_zones:
        by_zone = (
            merged.groupby(["pickup_location_id", "zone", "borough"], as_index=False)
            .agg(
                mae=("abs_error", "mean"),
                avg_actual=("rides", "mean"),
                avg_predicted=("predicted_demand", "mean"),
                hours=("pickup_hour", "nunique"),
            )
            .sort_values("mae", ascending=False)
        )
        by_zone["mae_pct_of_volume"] = by_zone["mae"] / by_zone["avg_actual"].clip(lower=1)
        st.dataframe(
            by_zone.head(25).rename(
                columns={
                    "pickup_location_id": "Zone ID",
                    "zone": "Zone",
                    "borough": "Borough",
                    "mae": "MAE",
                    "avg_actual": "Avg actual",
                    "avg_predicted": "Avg predicted",
                    "hours": "Hours",
                    "mae_pct_of_volume": "MAE / volume",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.1f"),
                "Avg actual": st.column_config.NumberColumn(format="%.1f"),
                "Avg predicted": st.column_config.NumberColumn(format="%.1f"),
                "MAE / volume": st.column_config.NumberColumn(format="%.0%"),
            },
        )

    with tab_zone_detail:
        zones = merged.groupby("zone")["rides"].sum().sort_values(ascending=False).index.tolist()
        chosen = st.selectbox("Zone", zones, index=0, key="monitor_zone")
        detail = merged[merged["zone"] == chosen].sort_values("pickup_hour")
        detail = detail.assign(pickup_hour=detail["pickup_hour_ny"])
        st.plotly_chart(
            plot_actual_vs_predicted(detail, title=f"{chosen} - actual vs predicted"),
            width="stretch",
            key="zone_detail",
        )
        st.caption(
            f"Zone MAE: {detail['abs_error'].mean():.2f} rides over {len(detail)} hours (times in {config.LOCAL_TIMEZONE})."
        )
