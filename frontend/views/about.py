"""About page: architecture, configuration and runtime status."""

from __future__ import annotations

import streamlit as st

import src.config as config
from frontend import data, ui

PIPELINE_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#94a3b8", fontname="Helvetica", fontsize=11];
  edge [color="#64748b"];
  tlc  [label="NYC TLC\\nyellow-taxi parquet", fillcolor="#fef3c7"];
  fp   [label="Feature pipeline\\n(hourly, GitHub Actions)\\nraw trips -> rides per zone per hour", fillcolor="#e0f2fe"];
  fg   [label="Hopsworks feature group\\ntime_series_hourly_feature_group_v4", fillcolor="#ede9fe"];
  fv   [label="Feature view", fillcolor="#ede9fe"];
  tp   [label="Training pipeline\\n(weekly)\\nLightGBM, hold-out MAE", fillcolor="#e0f2fe"];
  mr   [label="Model registry\\ntaxi_demand_predictor_next_hour", fillcolor="#ede9fe"];
  ip   [label="Inference pipeline\\n(hourly, after features)\\n672-h window per zone", fillcolor="#e0f2fe"];
  pg   [label="Predictions feature group\\ntaxi_hourly_model_prediction_v5", fillcolor="#ede9fe"];
  app  [label="Streamlit app\\nmap / tables / monitoring", fillcolor="#dcfce7"];
  tlc -> fp -> fg -> fv; fv -> tp -> mr; fv -> ip; mr -> ip -> pg -> app; fv -> app;
}
"""


def render() -> None:
    source, note = data.get_source()
    ui.hero(
        "About the pipeline",
        "A three-pipeline MLOps setup: features are computed hourly, the model is retrained weekly, and predictions are written hourly for the app to read.",
        [ui.mode_pill(source.mode)],
    )
    if note:
        st.warning(note, icon="⚠️")

    st.graphviz_chart(PIPELINE_DOT, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **How a prediction is made**
            1. Trips are aggregated into rides per TLC zone per hour (New-York time, stored as UTC).
            2. For each zone the previous 672 hours (28 days) become lag features `rides_t-672 … rides_t-1`.
            3. The pipeline adds the average of the same hour in the last four weeks plus hour-of-day and weekday.
            4. A LightGBM regressor predicts the number of pickups in the next hour.

            **Live vs demo**
            - *Live* reads features and predictions from Hopsworks (needs `HOPSWORKS_API_KEY`).
            - *Demo* replays real January 2024 traffic onto today's clock and runs the bundled model locally, so the app works anywhere.
            """
        )
    with c2:
        st.markdown("**Configuration**")
        st.table(
            {
                "setting": [
                    "Mode",
                    "Source",
                    "Feature group",
                    "Feature view",
                    "Predictions group",
                    "Model",
                    "Window",
                    "Timezone",
                ],
                "value": [
                    source.mode,
                    source.status().detail,
                    f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}",
                    f"{config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}",
                    f"{config.FEATURE_GROUP_MODEL_PREDICTION} v{config.FEATURE_GROUP_MODEL_PREDICTION_VERSION}",
                    config.MODEL_NAME,
                    f"{config.WINDOW_SIZE} hours",
                    config.LOCAL_TIMEZONE,
                ],
            }
        )

    st.markdown(
        """
        **Data note** - the TLC publishes trip files with a ~2-month delay. When the current month is not available yet, the
        feature pipeline replays the same month of the previous year, so the "live" numbers are a realistic simulation
        rather than real-time taxi activity.

        Source code: [github.com/VaibhavBansal26/nyc-taxi-ride-v3](https://github.com/VaibhavBansal26/nyc-taxi-ride-v3)
        """
    )
