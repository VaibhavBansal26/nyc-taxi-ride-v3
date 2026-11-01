"""scikit-learn pipeline used for training and inference.

Keep the class/function names and this module path stable: the serialised
model (``models/lgb_model.pkl`` and the Hopsworks registry copy) references
``src.pipeline_utils.TemporalFeatureEngineer`` and
``src.pipeline_utils.average_rides_last_4_weeks`` by import path.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

REQUIRED_LAGS_FOR_AVG_4W = [7 * 24, 14 * 24, 21 * 24, 28 * 24]


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """Add the mean of the same hour one to four weeks ago."""
    cols = [f"rides_t-{lag}" for lag in REQUIRED_LAGS_FOR_AVG_4W]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    X = X.copy()
    X["average_rides_last_4_weeks"] = X[cols].mean(axis=1)
    return X


add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate=False)


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive ``hour`` and ``day_of_week`` from ``pickup_hour`` and drop id columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if "pickup_hour" in X_.columns:
            if not pd.api.types.is_datetime64_any_dtype(X_["pickup_hour"]):
                X_["pickup_hour"] = pd.to_datetime(X_["pickup_hour"], errors="coerce", utc=True)
            X_["hour"] = X_["pickup_hour"].dt.hour
            X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        elif not {"hour", "day_of_week"}.issubset(X_.columns):
            raise ValueError("TemporalFeatureEngineer needs 'pickup_hour' or both 'hour' and 'day_of_week'.")
        return X_.drop(columns=[c for c in ("pickup_hour", "pickup_location_id") if c in X_.columns])


add_temporal_features = TemporalFeatureEngineer()


def ensure_required_lag_features(
    X: pd.DataFrame, feature_col: str = "rides", required_lags=None, fill_value: float = 0.0
) -> pd.DataFrame:
    """Guarantee the lag columns needed by :func:`average_rides_last_4_weeks` exist."""
    required_lags = REQUIRED_LAGS_FOR_AVG_4W if required_lags is None else required_lags
    X = X.copy()
    for lag in required_lags:
        col = f"{feature_col}_t-{lag}"
        if col not in X.columns:
            X[col] = fill_value
    return X


def get_pipeline(**hyper_params) -> Pipeline:
    """Feature engineering + LightGBM regressor."""
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),
    )
