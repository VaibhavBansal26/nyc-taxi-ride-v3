# import lightgbm as lgb
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import FunctionTransformer


# # Function to calculate the average rides over the last 4 weeks
# def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
#     last_4_weeks_columns = [
#         f"rides_t-{7*24}",  # 1 week ago
#         f"rides_t-{14*24}",  # 2 weeks ago
#         f"rides_t-{21*24}",  # 3 weeks ago
#         f"rides_t-{28*24}",  # 4 weeks ago
#     ]

#     # Ensure the required columns exist in the DataFrame
#     for col in last_4_weeks_columns:
#         if col not in X.columns:
#             raise ValueError(f"Missing required column: {col}")

#     # Calculate the average of the last 4 weeks
#     X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

#     return X


# # FunctionTransformer to add the average rides feature
# add_feature_average_rides_last_4_weeks = FunctionTransformer(
#     average_rides_last_4_weeks, validate=False
# )


# # Custom transformer to add temporal features
# class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X_ = X.copy()
#         X_["hour"] = X_["pickup_hour"].dt.hour
#         X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

#         return X_.drop(columns=["pickup_hour", "pickup_location_id"])


# # Instantiate the temporal feature engineer
# add_temporal_features = TemporalFeatureEngineer()


# # Function to return the pipeline
# def get_pipeline(**hyper_params):
#     """
#     Returns a pipeline with optional parameters for LGBMRegressor.

#     Parameters:
#     ----------
#     **hyper_params : dict
#         Optional parameters to pass to the LGBMRegressor.

#     Returns:
#     -------
#     pipeline : sklearn.pipeline.Pipeline
#         A pipeline with feature engineering and LGBMRegressor.
#     """
#     pipeline = make_pipeline(
#         add_feature_average_rides_last_4_weeks,
#         add_temporal_features,
#         lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
#     )
#     return pipeline

import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# ---- weekly average feature ----
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4 = [f"rides_t-{7*24}", f"rides_t-{14*24}", f"rides_t-{21*24}", f"rides_t-{28*24}"]
    for col in last_4:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")
    X = X.copy()
    X["average_rides_last_4_weeks"] = X[last_4].mean(axis=1)
    return X

add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)

# ---- temporal features ----
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        import pandas as pd
        X_ = X.copy()
        if "pickup_hour" in X_.columns:
            if not pd.api.types.is_datetime64_any_dtype(X_["pickup_hour"]):
                X_["pickup_hour"] = pd.to_datetime(X_["pickup_hour"], errors="coerce", utc=True)
            X_["hour"] = X_["pickup_hour"].dt.hour
            X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        elif {"hour", "day_of_week"}.issubset(X_.columns):
            pass
        else:
            sample = list(X_.columns)[:30]
            raise ValueError(
                "TemporalFeatureEngineer: 'pickup_hour' missing and 'hour/day_of_week' not present. "
                f"Columns: {sample}"
            )
        drop_cols = [c for c in ["pickup_hour", "pickup_location_id"] if c in X_.columns]
        return X_.drop(columns=drop_cols)

add_temporal_features = TemporalFeatureEngineer()

# ---- inference guard ----
REQUIRED_LAGS_FOR_AVG_4W = [168, 336, 504, 672]

def ensure_required_lag_features(
    X: pd.DataFrame, feature_col: str = "rides", required_lags=None, fill_value: float = 0.0
) -> pd.DataFrame:
    if required_lags is None:
        required_lags = REQUIRED_LAGS_FOR_AVG_4W
    X = X.copy()
    for lag in required_lags:
        col = f"{feature_col}_t-{lag}"
        if col not in X.columns:
            X[col] = fill_value
    return X

# ---- pipeline factory ----
def get_pipeline(**hyper_params):
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params),
    )
