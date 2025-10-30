# pipelines/ensure_feature_view.py
import pandas as pd, hopsworks
import src.config as config
from src.inference import _ensure_feature_view

project = hopsworks.login(project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
fv = _ensure_feature_view(fs)
print(f"✅ Feature View ready: {fv.name} v{fv.version}")
start = pd.Timestamp.utcnow() - pd.Timedelta(days=1)
end = pd.Timestamp.utcnow()
df = fv.get_batch_data(start_time=start, end_time=end)
print(f"Read {len(df)} rows.")
