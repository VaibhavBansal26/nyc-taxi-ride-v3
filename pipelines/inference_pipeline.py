"""Hourly inference pipeline: feature store -> model -> predictions feature group.

Run with ``python -m pipelines.inference_pipeline`` (``--dry-run`` to skip the write,
``--local-model`` to use ``models/lgb_model.pkl`` instead of the registry).
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.inference import (
    get_model_predictions,
    load_backfill_features,
    load_batch_of_features_from_store,
    load_local_model,
    load_model_from_registry,
    write_predictions,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def run(dry_run: bool = False, local_model: bool = False, backfill_hours: int = 0) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    if backfill_hours > 0:
        features = load_backfill_features(backfill_hours, now)
        log.info(
            "Built %d feature rows for %d target hours (backfill)", len(features), features["pickup_hour"].nunique()
        )
    else:
        features = load_batch_of_features_from_store(now)
        log.info("Built %d feature rows (one per zone) for %s", len(features), features["pickup_hour"].iloc[0])

    model = load_local_model() if local_model else load_model_from_registry()
    preds = get_model_predictions(model, features)
    log.info(
        "Generated %d predictions over %d target hour(s); %.0f predicted rides in the latest hour",
        len(preds),
        preds["pickup_hour"].nunique(),
        preds.loc[preds["pickup_hour"] == preds["pickup_hour"].max(), "predicted_demand"].sum(),
    )

    if dry_run:
        log.info("Dry run - not writing. Top zones:\n%s", preds.nlargest(5, "predicted_demand").to_string(index=False))
    else:
        write_predictions(preds)
        log.info("Prediction insert submitted.")
    return preds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict next-hour taxi demand per zone")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-model", action="store_true", help="use models/lgb_model.pkl instead of the registry")
    parser.add_argument(
        "--backfill-hours",
        type=int,
        default=0,
        help="also (re)generate predictions for the last N hours, e.g. after an outage",
    )
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run, local_model=args.local_model, backfill_hours=args.backfill_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
