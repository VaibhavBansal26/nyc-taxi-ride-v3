"""Plotly figures shared by the Streamlit app and notebooks."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import LOCAL_TIMEZONE

PRIMARY = "#0ea5e9"
ACCENT = "#ef4444"
MUTED = "#94a3b8"


def _lag_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("rides_t-")]
    return sorted(cols, key=lambda c: -int(c.split("t-")[1]))


def plot_prediction(
    features: pd.DataFrame, predicted_demand: float, title: str | None = None, hours: int = 24 * 7
) -> go.Figure:
    """History of one zone (last ``hours`` lags) with the predicted next hour.

    ``features`` must contain exactly one row in the model feature layout.
    """
    if features is None or features.empty:
        raise ValueError("plot_prediction needs one feature row")
    row = features.iloc[0]
    lag_cols = _lag_columns(features)[-hours:]
    pickup_hour = pd.Timestamp(row["pickup_hour"])
    if pickup_hour.tzinfo is None:
        pickup_hour = pickup_hour.tz_localize("UTC")
    pickup_hour = pickup_hour.tz_convert(LOCAL_TIMEZONE)

    history = pd.DataFrame(
        {
            "datetime": pd.date_range(end=pickup_hour - pd.Timedelta(hours=1), periods=len(lag_cols), freq="h"),
            "rides": [float(row[c]) for c in lag_cols],
        }
    )
    title = title or f"Location {row['pickup_location_id']} - predicting {pickup_hour:%a %d %b %H:%M}"
    fig = px.line(
        history,
        x="datetime",
        y="rides",
        template="plotly_white",
        title=title,
        labels={"datetime": "Hour (NY)", "rides": "Rides"},
    )
    fig.update_traces(line_color=PRIMARY, name="Observed", showlegend=True)
    fig.add_scatter(
        x=[pickup_hour],
        y=[float(predicted_demand)],
        mode="markers",
        marker=dict(symbol="x", size=14, color=ACCENT),
        name="Prediction",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.1), height=320)
    return fig


def plot_mae_by_hour(mae_df: pd.DataFrame) -> go.Figure:
    """``mae_df`` columns: pickup_hour, mae."""
    fig = px.line(
        mae_df,
        x="pickup_hour",
        y="mae",
        markers=True,
        template="plotly_white",
        labels={"pickup_hour": "Hour (NY)", "mae": "MAE (rides)"},
    )
    fig.update_traces(line_color=PRIMARY)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
    return fig


def plot_actual_vs_predicted(df: pd.DataFrame, title: str = "") -> go.Figure:
    """``df`` columns: pickup_hour, rides, predicted_demand."""
    fig = go.Figure()
    fig.add_scatter(x=df["pickup_hour"], y=df["rides"], mode="lines+markers", name="Actual", line=dict(color=PRIMARY))
    fig.add_scatter(
        x=df["pickup_hour"],
        y=df["predicted_demand"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color=ACCENT, dash="dot"),
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Hour (NY)",
        yaxis_title="Rides",
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_top_zones(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bar chart of the busiest zones. ``df`` columns: zone, borough, predicted_demand."""
    top = df.nlargest(n, "predicted_demand").sort_values("predicted_demand")
    fig = px.bar(
        top,
        x="predicted_demand",
        y="zone",
        color="borough",
        orientation="h",
        template="plotly_white",
        labels={"predicted_demand": "Predicted rides", "zone": "", "borough": "Borough"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=max(320, 26 * len(top)), legend=dict(orientation="h", y=-0.15)
    )
    return fig


def plot_borough_share(df: pd.DataFrame) -> go.Figure:
    share = (
        df.groupby("borough", as_index=False)["predicted_demand"].sum().sort_values("predicted_demand", ascending=False)
    )
    fig = px.pie(share, names="borough", values="predicted_demand", hole=0.55, template="plotly_white")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, legend=dict(orientation="h", y=-0.1))
    return fig
