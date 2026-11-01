"""Weekly retraining: fetch recent rides, train, evaluate on a hold-out, register if better.

Run with ``python -m pipelines.model_training_pipeline`` (``--dry-run`` skips registration).
"""

from __future__ import annotations

import argparse
import logging
import sys

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target
from src.inference import fetch_days_data, get_hopsworks_project, load_metrics_from_registry
from src.pipeline_utils import get_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

HYPER_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "verbose": -1}


def time_split(features: pd.DataFrame, targets: pd.Series, holdout_days: int = 7):
    """Last ``holdout_days`` of target hours form the validation set."""
    cutoff = pd.to_datetime(features["pickup_hour"]).max() - pd.Timedelta(days=holdout_days)
    train_mask = pd.to_datetime(features["pickup_hour"]) < cutoff
    return features[train_mask], targets[train_mask], features[~train_mask], targets[~train_mask]


def train_and_evaluate(ts_data: pd.DataFrame, step_size: int = 23):
    features, targets = transform_ts_data_info_features_and_target(
        ts_data, window_size=config.WINDOW_SIZE, step_size=step_size
    )
    if features.empty:
        raise ValueError("Not enough contiguous data to build training windows")
    X_train, y_train, X_val, y_val = time_split(features, targets)
    log.info("Training on %d windows, validating on %d", len(X_train), len(X_val))

    pipeline = get_pipeline(**HYPER_PARAMS)
    pipeline.fit(X_train, y_train)
    val_mae = mean_absolute_error(y_val, pipeline.predict(X_val)) if len(X_val) else float("nan")

    # refit on everything before registering
    pipeline = get_pipeline(**HYPER_PARAMS).fit(features, targets)
    return pipeline, val_mae, features, targets


def register(pipeline, mae: float, features: pd.DataFrame, targets: pd.Series) -> None:
    from hsml.model_schema import ModelSchema
    from hsml.schema import Schema

    model_path = config.MODELS_DIR / "lgb_model.pkl"
    joblib.dump(pipeline, model_path)
    model_schema = ModelSchema(input_schema=Schema(features), output_schema=Schema(targets.to_frame()))
    registry = get_hopsworks_project().get_model_registry()
    model = registry.sklearn.create_model(
        name=config.MODEL_NAME,
        metrics={"test_mae": float(mae)},
        input_example=features.sample(1, random_state=0),
        model_schema=model_schema,
        description="LightGBM next-hour taxi demand model",
    )
    model.save(str(model_path))
    log.info("Registered %s version %s", config.MODEL_NAME, model.version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain the next-hour demand model")
    parser.add_argument("--dry-run", action="store_true", help="train and evaluate but never register")
    parser.add_argument("--days", type=int, default=config.TRAINING_LOOKBACK_DAYS)
    parser.add_argument("--force", action="store_true", help="register even if not better than the current model")
    args = parser.parse_args(argv)

    log.info("Fetching %d days of hourly rides ...", args.days)
    ts_data = fetch_days_data(args.days)
    log.info("%d rows", len(ts_data))

    pipeline, val_mae, features, targets = train_and_evaluate(ts_data)
    previous = load_metrics_from_registry().get("test_mae")
    log.info(
        "New hold-out MAE: %.4f | previous registered MAE: %s",
        val_mae,
        f"{previous:.4f}" if previous is not None else "none",
    )

    if args.dry_run:
        log.info("Dry run - not registering")
        return 0
    if args.force or previous is None or val_mae < previous:
        register(pipeline, val_mae, features, targets)
    else:
        log.info("New model is not better; keeping the current one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
