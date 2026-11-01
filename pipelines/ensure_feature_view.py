"""One-off helper: make sure the feature view exists and read a day of data."""

from __future__ import annotations

import pandas as pd

from src.inference import ensure_feature_view, get_feature_store, read_hourly_rides


def main() -> int:
    fs = get_feature_store()
    fv = ensure_feature_view(fs)
    print(f"Feature view ready: {fv.name} v{fv.version}")
    end = pd.Timestamp.now(tz="UTC")
    df = read_hourly_rides(end - pd.Timedelta(days=1), end, fs)
    print(f"Read {len(df)} rows for the last 24 h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
