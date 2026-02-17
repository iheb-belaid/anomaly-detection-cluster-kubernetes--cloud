import logging
from typing import List, Dict, Any, Set

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

from app import config
from app.ml.anomaly_detector import AnomalyDetector, PodAnomalyResult
from app.prometheus.api_client import PrometheusAPI, PrometheusError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("anomaly-service")

app = FastAPI(title="K8s Anomaly Detection Service")

prom_client = PrometheusAPI()
detector = AnomalyDetector(prom_client)

# Prometheus metrics
anomaly_flag_gauge = Gauge(
    "anomaly_flag", "Anomaly flag per pod (1=anomaly,0=normal)", ["pod"]
)
anomaly_score_gauge = Gauge(
    "anomaly_score", "Anomaly score per pod (IsolationForest decision function)", ["pod"]
)

_known_pods: Set[str] = set()


def _update_prometheus_metrics(results: List[PodAnomalyResult]) -> None:
    """
    Update Prometheus Gauges for anomaly flag and score for each pod.

    This function also removes metric series for pods that are no longer present
    in the latest detection results in order to avoid leaking stale labels.

    Parameters
    ----------
    results:
        List of :class:`PodAnomalyResult` instances produced by the detector.
    """
    global _known_pods
    current_pods = set()

    for r in results:
        current_pods.add(r.pod)
        anomaly_flag_gauge.labels(pod=r.pod).set(r.anomaly_flag)
        anomaly_score_gauge.labels(pod=r.pod).set(r.anomaly_score)

    # Remove metrics for pods that disappeared
    pods_to_remove = _known_pods - current_pods
    for pod in pods_to_remove:
        anomaly_flag_gauge.remove(pod=pod)
        anomaly_score_gauge.remove(pod=pod)

    _known_pods = current_pods


@app.on_event("startup")
def startup_event() -> None:
    """
    Application startup hook that trains the anomaly detection model.

    The service attempts to train the model from historical metrics during startup.
    If this fails (for example because Prometheus is temporarily unavailable),
    the service will still start but ``/detect`` will return HTTP 503 until
    training is successfully completed.
    """
    logger.info("Service startup: training anomaly detection model")
    try:
        detector.train()
        logger.info("Initial model training succeeded")
    except (PrometheusError, RuntimeError) as exc:
        logger.exception("Initial model training failed: %s", exc)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns a small JSON payload containing a status flag, whether the model
    is currently trained, and the configured Prometheus URL.
    """
    return {
        "status": "ok",
        "model_trained": detector.is_trained,
        "prometheus_url": config.PROMETHEUS_URL,
    }


@app.get("/detect")
def detect() -> Dict[str, Any]:
    """
    Run anomaly detection on current metrics.

    Queries Prometheus for the latest CPU, memory, and restart metrics, runs the
    anomaly detector, updates the exported Prometheus Gauges, and returns a JSON
    payload with anomaly flags and scores per pod.

    Returns
    -------
    dict
        JSON structure of the form ``{\"pods\": [...]}`` where each item describes
        a single pod and its associated metrics and anomaly outputs.

    Raises
    ------
    HTTPException
        * 503 if the model is not trained or Prometheus cannot be queried.
        * 500 if an unexpected runtime error occurs during detection.
    """
    if not detector.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet; check logs for training errors.",
        )

    try:
        results = detector.detect_current()
    except PrometheusError as exc:
        logger.exception("Failed to query Prometheus during detection: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to query Prometheus for current metrics.",
        ) from exc
    except RuntimeError as exc:
        logger.exception("Detection failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    _update_prometheus_metrics(results)

    payload = [
        {
            "pod": r.pod,
            "cpu": r.cpu,
            "memory": r.memory,
            "restarts": r.restarts,
            "anomaly_flag": r.anomaly_flag,
            "anomaly_score": r.anomaly_score,
        }
        for r in results
    ]

    return {"pods": payload}


@app.get("/metrics")
def metrics() -> Response:
    """
    Prometheus scrape endpoint exposing anomaly metrics.

    Returns the text exposition format produced by :mod:`prometheus_client`,
    including the ``anomaly_flag`` and ``anomaly_score`` Gauges for each pod.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
