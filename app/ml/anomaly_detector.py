from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app import config
from app.prometheus.api_client import PrometheusAPI, PrometheusError

logger = logging.getLogger(__name__)


@dataclass
class PodAnomalyResult:
    """
    Container for per-pod anomaly detection outputs and raw features.

    Attributes
    ----------
    pod:
        Pod name as taken from the Prometheus label configured via ``POD_LABEL``.
    cpu:
        CPU usage feature value for the pod at a given point in time.
    memory:
        Memory usage feature value for the pod at a given point in time.
    restarts:
        Restart count feature value for the pod at a given point in time.
    anomaly_flag:
        Binary flag indicating whether the observation is anomalous
        (``1`` for anomaly, ``0`` for normal) according to IsolationForest.
    anomaly_score:
        IsolationForest decision function value; lower typically means "more anomalous".
    """
    pod: str
    cpu: float
    memory: float
    restarts: float
    anomaly_flag: int  # 1 = anomaly, 0 = normal
    anomaly_score: float


class AnomalyDetector:
    """
    Perform anomaly detection over Kubernetes pod metrics from Prometheus.

    This class encapsulates:

    * Fetching historical and real-time metrics via :class:`PrometheusAPI`.
    * Building feature vectors ``[cpu, memory, restarts]``.
    * Fitting a :class:`sklearn.preprocessing.StandardScaler` and
      :class:`sklearn.ensemble.IsolationForest` model.
    * Producing :class:`PodAnomalyResult` objects for the current state of all pods.
    """

    def __init__(self, prom_client: PrometheusAPI) -> None:
        """
        Initialize the anomaly detector and PromQL queries.

        Parameters
        ----------
        prom_client:
            Instance of :class:`PrometheusAPI` used to communicate with Prometheus.
        """
        self.prom_client = prom_client
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=config.IFOREST_N_ESTIMATORS,
            max_samples=config.IFOREST_MAX_SAMPLES,
            contamination=config.IFOREST_CONTAMINATION,
            random_state=42,
        )
        self.is_trained = False
        self._lock = threading.Lock()

        # PromQL templates with namespace regex filter
        ns_filter = f',namespace=~"{config.NAMESPACE_REGEX}"'
        self.cpu_query = (
            'rate(container_cpu_usage_seconds_total{container!="",container!="POD"'
            f"{ns_filter}"
            '}[1m])'
        )
        self.mem_query = (
            'container_memory_usage_bytes{container!="",container!="POD"'
            f"{ns_filter}"
            "}"
        )
        self.restarts_query = (
            f'kube_pod_container_status_restarts_total{{namespace=~"{config.NAMESPACE_REGEX}"}}'
        )

    def train(self) -> None:
        """
        Train the scaler and IsolationForest model on historical metrics.

        The method pulls the last ``TRAINING_LOOKBACK_MINUTES`` of metrics from Prometheus,
        builds aligned feature vectors, and fits the internal scaler and IsolationForest model.

        Raises
        ------
        RuntimeError
            If an insufficient number of samples is available to fit the model.
        PrometheusError
            If Prometheus is unreachable or returns an invalid response.
        """
        with self._lock:
            logger.info("Starting model training from Prometheus range data")
            X = self._fetch_training_dataset()

            if X.size == 0 or X.shape[0] < 10:
                raise RuntimeError(
                    f"Not enough training samples ({X.shape[0]}) to fit IsolationForest"
                )

            logger.info("Fitting scaler and IsolationForest on %d samples", X.shape[0])
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled)
            self.is_trained = True
            logger.info("Model training completed")

    def detect_current(self) -> List[PodAnomalyResult]:
        """
        Run anomaly detection on the most recent metrics.

        This method performs instant queries for CPU, memory, and restart metrics,
        aligns them per pod, applies the fitted scaler and IsolationForest model,
        and returns anomaly scores and flags.

        Returns
        -------
        list of PodAnomalyResult
            One result per pod that has all required metrics.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        PrometheusError
            If Prometheus is unreachable or returns an invalid response.
        """
        with self._lock:
            if not self.is_trained:
                raise RuntimeError("Model is not trained yet")

            logger.debug("Fetching current metrics for anomaly detection")
            cpu_result = self.prom_client.query(self.cpu_query)
            mem_result = self.prom_client.query(self.mem_query)
            restart_result = self.prom_client.query(self.restarts_query)

            cpu_by_pod = self._extract_instant_by_pod(cpu_result)
            mem_by_pod = self._extract_instant_by_pod(mem_result)
            restarts_by_pod = self._extract_instant_by_pod(restart_result)

            common_pods = (
                set(cpu_by_pod.keys())
                & set(mem_by_pod.keys())
                & set(restarts_by_pod.keys())
            )

            if not common_pods:
                logger.warning(
                    "No pods have all three metrics (cpu, memory, restarts) at current time"
                )
                return []

            pods_sorted = sorted(common_pods)
            features = []
            for pod in pods_sorted:
                features.append(
                    [
                        cpu_by_pod[pod],
                        mem_by_pod[pod],
                        restarts_by_pod[pod],
                    ]
                )

            X = np.array(features, dtype=float)
            X_scaled = self.scaler.transform(X)
            scores = self.model.decision_function(X_scaled)
            preds = self.model.predict(X_scaled)  # 1 for normal, -1 for anomaly

            results: List[PodAnomalyResult] = []
            for pod, score, pred in zip(pods_sorted, scores, preds):
                flag = 1 if pred == -1 else 0
                results.append(
                    PodAnomalyResult(
                        pod=pod,
                        cpu=cpu_by_pod[pod],
                        memory=mem_by_pod[pod],
                        restarts=restarts_by_pod[pod],
                        anomaly_flag=flag,
                        anomaly_score=float(score),
                    )
                )

            logger.info("Computed anomaly scores for %d pods", len(results))
            return results

    def _fetch_training_dataset(self) -> np.ndarray:
        """
        Build the training dataset from Prometheus range queries.

        This helper calls ``/api/v1/query_range`` for CPU, memory and restarts, then
        aligns the resulting time series for pods that are present in all three metrics.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array of shape ``(n_samples, 3)`` containing
            ``[cpu, memory, restarts]`` feature vectors. If no aligned data is
            available, an empty array with shape ``(0, 3)`` is returned.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=config.TRAINING_LOOKBACK_MINUTES)
        step_str = f"{config.QUERY_STEP_SECONDS}s"

        logger.info(
            "Fetching training data from %s to %s (step=%s)",
            start.isoformat(),
            now.isoformat(),
            step_str,
        )

        cpu_result = self.prom_client.query_range(self.cpu_query, start, now, step_str)
        mem_result = self.prom_client.query_range(self.mem_query, start, now, step_str)
        restart_result = self.prom_client.query_range(
            self.restarts_query, start, now, step_str
        )

        cpu_by_pod = self._extract_range_by_pod(cpu_result)
        mem_by_pod = self._extract_range_by_pod(mem_result)
        restarts_by_pod = self._extract_range_by_pod(restart_result)

        common_pods = (
            set(cpu_by_pod.keys())
            & set(mem_by_pod.keys())
            & set(restarts_by_pod.keys())
        )

        if not common_pods:
            logger.warning(
                "No pods have all three metrics across the training window; training dataset will be empty"
            )
            return np.empty((0, 3), dtype=float)

        X: List[List[float]] = []
        for pod in common_pods:
            cpu_series = cpu_by_pod[pod]
            mem_series = mem_by_pod[pod]
            restart_series = restarts_by_pod[pod]
            length = min(len(cpu_series), len(mem_series), len(restart_series))
            for i in range(length):
                X.append(
                    [
                        cpu_series[i],
                        mem_series[i],
                        restart_series[i],
                    ]
                )

        if not X:
            logger.warning("No aligned samples were built for training")
            return np.empty((0, 3), dtype=float)

        return np.array(X, dtype=float)

    @staticmethod
    def _extract_range_by_pod(result: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Convert a Prometheus range query result into a mapping by pod.

        Parameters
        ----------
        result:
            Raw ``data.result`` list from a ``/api/v1/query_range`` response.

        Returns
        -------
        dict
            Mapping of pod name to a list of float samples ordered by time.
        """
        by_pod: Dict[str, List[float]] = {}
        for ts in result:
            metric = ts.get("metric", {})
            pod = metric.get(config.POD_LABEL)
            if not pod:
                continue

            values = ts.get("values", [])
            series: List[float] = []
            for _, value_str in values:
                try:
                    series.append(float(value_str))
                except (TypeError, ValueError):
                    continue

            if series:
                by_pod[pod] = series

        return by_pod

    @staticmethod
    def _extract_instant_by_pod(result: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Convert an instant Prometheus query result into a mapping by pod.

        Parameters
        ----------
        result:
            Raw ``data.result`` list from an ``/api/v1/query`` response.

        Returns
        -------
        dict
            Mapping of pod name to a single float value for the current timestamp.
        """
        by_pod: Dict[str, float] = {}
        for ts in result:
            metric = ts.get("metric", {})
            pod = metric.get(config.POD_LABEL)
            if not pod:
                continue

            value_pair = ts.get("value")
            if not value_pair or len(value_pair) != 2:
                continue

            _, value_str = value_pair
            try:
                value = float(value_str)
            except (TypeError, ValueError):
                continue

            by_pod[pod] = value

        return by_pod
