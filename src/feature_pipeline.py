"""Hourly feature pipeline: raw TLC trips -> hourly rides -> Hopsworks feature group.

Run with ``python -m src.feature_pipeline``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import src.config as config
from src.data_utils import fetch_batch_raw_data, transform_raw_data_into_ts_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def build_hourly_rides(
    now: pd.Timestamp | None = None, lookback_days: int = config.FEATURE_PIPELINE_LOOKBACK_DAYS
) -> pd.DataFrame:
    current = (now or pd.Timestamp.now(tz="UTC")).ceil("h")
    fetch_from = current - pd.Timedelta(days=lookback_days)
    log.info("Fetching raw trips %s .. %s", fetch_from, current)
    rides = fetch_batch_raw_data(fetch_from, current)
    log.info("Raw trips: %d rows", len(rides))
    ts_data = transform_raw_data_into_ts_data(rides)
    log.info(
        "Hourly rides: %d rows over %d zones",
        len(ts_data),
        ts_data["pickup_location_id"].nunique() if len(ts_data) else 0,
    )
    return ts_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute hourly rides and push them to Hopsworks")
    parser.add_argument("--dry-run", action="store_true", help="build features but do not write to Hopsworks")
    parser.add_argument("--lookback-days", type=int, default=config.FEATURE_PIPELINE_LOOKBACK_DAYS)
    parser.add_argument("--wait", action="store_true", help="wait for the Hopsworks materialisation job to finish")
    args = parser.parse_args(argv)

    ts_data = build_hourly_rides(lookback_days=args.lookback_days)
    if ts_data.empty:
        log.error("No hourly rides produced; aborting")
        return 1
    if args.dry_run:
        log.info("Dry run - skipping Hopsworks insert. Sample:\n%s", ts_data.tail().to_string())
        return 0

    from src.inference import ensure_feature_view, get_feature_store, get_rides_feature_group

    fs = get_feature_store()
    fg = get_rides_feature_group(fs, create=True)
    log.info("Inserting into %s v%s ...", config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION)
    fg.insert(ts_data, write_options={"wait_for_job": args.wait})
    log.info("Insert %s.", "materialised" if args.wait else "submitted")
    try:
        ensure_feature_view(fs)
        log.info("Feature view %s v%s ready", config.FEATURE_VIEW_NAME, config.FEATURE_VIEW_VERSION)
    except Exception as exc:
        log.warning("Could not ensure feature view: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
