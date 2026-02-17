import os

"""
Basic configuration for the anomaly detection service.

Values can be overridden via environment variables.
"""

# Base URL for the Prometheus HTTP API (no trailing slash)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

# Regex for namespaces to include in queries, e.g. "default|prod-.*"
NAMESPACE_REGEX = os.getenv("NAMESPACE_REGEX", ".*")

# PromQL label used for pod name (commonly "pod")
POD_LABEL = os.getenv("POD_LABEL", "pod")

# Lookback window (in minutes) for model training
TRAINING_LOOKBACK_MINUTES = int(os.getenv("TRAINING_LOOKBACK_MINUTES", "60"))

# Step (in seconds) for range queries during training
QUERY_STEP_SECONDS = int(os.getenv("QUERY_STEP_SECONDS", "60"))

# Request timeout for Prometheus HTTP calls (seconds)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))

# IsolationForest parameters
IFOREST_N_ESTIMATORS = int(os.getenv("IFOREST_N_ESTIMATORS", "100"))
_max_samples_env = os.getenv("IFOREST_MAX_SAMPLES", "256")
IFOREST_MAX_SAMPLES = (
    "auto" if _max_samples_env.lower() == "auto" else int(_max_samples_env)
)
IFOREST_CONTAMINATION = os.getenv("IFOREST_CONTAMINATION", "auto")
