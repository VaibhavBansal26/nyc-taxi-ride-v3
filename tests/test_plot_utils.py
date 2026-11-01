import pandas as pd

from src.plot_utils import plot_actual_vs_predicted, plot_mae_by_hour, plot_prediction, plot_top_zones


def test_plot_prediction_builds_figure():
    row = {f"rides_t-{k}": float(k) for k in range(672, 0, -1)}
    row.update({"pickup_location_id": 7, "pickup_hour": pd.Timestamp("2024-06-01 12:00", tz="UTC")})
    fig = plot_prediction(pd.DataFrame([row]), 12.0, hours=48)
    assert len(fig.data) == 2
    assert len(fig.data[0].x) == 48


def test_other_plots():
    hours = pd.date_range("2024-06-01", periods=5, freq="h", tz="America/New_York")
    assert plot_mae_by_hour(pd.DataFrame({"pickup_hour": hours, "mae": [1, 2, 3, 2, 1]})).data
    assert plot_actual_vs_predicted(
        pd.DataFrame({"pickup_hour": hours, "rides": [1, 2, 3, 4, 5], "predicted_demand": [1, 2, 2, 4, 6]})
    ).data
    assert plot_top_zones(
        pd.DataFrame({"zone": list("abc"), "borough": ["M", "M", "Q"], "predicted_demand": [3, 2, 1]})
    ).data
