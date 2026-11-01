"""Streamlit entry point - `streamlit run frontend/app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend.ui import inject_css, render_sidebar  # noqa: E402
from frontend.views import about, monitoring, predictions  # noqa: E402

st.set_page_config(
    page_title="NYC Taxi Demand - Next Hour",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

navigation = st.navigation(
    [
        st.Page(predictions.render, title="Next-hour demand", icon="🗺️", default=True),
        st.Page(monitoring.render, title="Model monitoring", icon="📈", url_path="monitoring"),
        st.Page(about.render, title="About the pipeline", icon="ℹ️", url_path="about"),
    ]
)
render_sidebar()
navigation.run()
