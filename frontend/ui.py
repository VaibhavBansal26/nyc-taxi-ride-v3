"""Shared UI pieces: CSS, hero banner, sidebar."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

import src.config as config
from frontend import data

CSS = """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  .hero {
    border-radius: 18px; padding: 22px 26px; color: #fff;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #f59e0b 140%);
    box-shadow: 0 10px 28px rgba(2, 6, 23, .18); margin-bottom: 1rem;
  }
  .hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 750; letter-spacing: .2px; color: #fff; }
  .hero p { margin: 0 0 10px 0; opacity: .95; font-size: 1rem; }
  .pill {
    display: inline-block; padding: 4px 11px; border-radius: 999px; margin-right: 8px; margin-top: 4px;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); font-size: 12px; color: #fff;
  }
  .pill.demo { background: rgba(245, 158, 11, .35); }
  .pill.live { background: rgba(34, 197, 94, .35); }
  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid rgba(2,6,23,.08); border-radius: 14px; padding: 12px 16px;
    box-shadow: 0 4px 18px rgba(2, 6, 23, .05);
  }
  button[role="tab"] { font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills: list[tuple[str, str]]) -> None:
    pill_html = "".join(f'<span class="pill {cls}">{text}</span>' for text, cls in pills)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pill_html}</div>', unsafe_allow_html=True)


def mode_pill(mode: str) -> tuple[str, str]:
    return ("● LIVE · Hopsworks" if mode == "live" else "● DEMO · offline replay", mode)


def ny_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=config.LOCAL_TIMEZONE)


@st.fragment(run_every=timedelta(seconds=60))
def _auto_refresh_tick() -> None:
    last = st.session_state.get("_last_auto_refresh")
    now = pd.Timestamp.now(tz="UTC")
    if last is not None and (now - last) >= pd.Timedelta(seconds=55):
        data.clear_prediction_caches()
        st.session_state["_last_auto_refresh"] = now
        st.rerun(scope="app")
    if last is None:
        st.session_state["_last_auto_refresh"] = now
    st.caption(f"Auto-refresh on · last {now.tz_convert(config.LOCAL_TIMEZONE):%H:%M:%S} NY")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🚕 NYC Taxi Demand")
        st.caption("Next-hour ride forecasts per taxi zone")

        options = ["auto", "live", "demo"]
        current = st.session_state.get("app_mode", config.APP_MODE if config.APP_MODE in options else "auto")
        mode = st.radio(
            "Data source",
            options,
            index=options.index(current),
            format_func=lambda m: {
                "auto": "Auto (live if available)",
                "live": "Live · Hopsworks",
                "demo": "Demo · offline",
            }[m],
            help="Live needs HOPSWORKS_API_KEY. Demo replays January 2024 through the bundled model.",
        )
        if mode != current:
            st.session_state["app_mode"] = mode
            data.clear_prediction_caches()
            st.rerun()
        st.session_state.setdefault("app_mode", mode)

        if st.button("🔄 Refresh data", width="stretch"):
            data.clear_prediction_caches()
            st.rerun()
        if st.toggle("Auto-refresh every minute", value=False):
            _auto_refresh_tick()

        st.divider()
        st.caption(f"NY time: {ny_now():%a %d %b %Y, %H:%M}")
        st.caption(f"Model window: {config.WINDOW_SIZE} h · cache TTL: {data.PREDICTION_TTL // 60} min")
