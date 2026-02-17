from datetime import datetime
from typing import List, Dict, Any

import requests
from requests import RequestException

from app import config


class PrometheusError(Exception):
    """
    Exception raised when Prometheus HTTP API calls fail or return invalid data.

    This is used to wrap lower-level :mod:`requests` exceptions as well as
    unexpected HTTP status codes or response payloads from Prometheus.
    """


class PrometheusAPI:
    """
    Minimal wrapper around the Prometheus HTTP API.

    The client exposes thin helper methods for:

    * ``/api/v1/query`` via :meth:`query`
    * ``/api/v1/query_range`` via :meth:`query_range`

    Both methods return the parsed ``data.result`` list from the Prometheus
    JSON response or raise :class:`PrometheusError` on error.
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        """
        Create a new Prometheus API client.

        Parameters
        ----------
        base_url:
            Base URL of the Prometheus server (e.g. ``\"http://prometheus:9090\"``).
            If omitted, :data:`app.config.PROMETHEUS_URL` is used.
        timeout:
            Request timeout in seconds. If omitted, :data:`app.config.REQUEST_TIMEOUT_SECONDS`
            is used.
        """
        self.base_url = (base_url or config.PROMETHEUS_URL).rstrip("/")
        self.timeout = timeout or config.REQUEST_TIMEOUT_SECONDS

    def _handle_response(self, resp: requests.Response) -> List[Dict[str, Any]]:
        """
        Validate and parse a Prometheus HTTP response.

        Parameters
        ----------
        resp:
            Response object returned by :func:`requests.get`.

        Returns
        -------
        list of dict
            The ``data.result`` portion of the Prometheus response.

        Raises
        ------
        PrometheusError
            If the HTTP status code is non-200, the body cannot be decoded as JSON,
            or the payload status is not ``\"success\"``.
        """
        if resp.status_code != 200:
            raise PrometheusError(f"Prometheus HTTP {resp.status_code}: {resp.text}")

        try:
            payload = resp.json()
        except ValueError as exc:  # JSON decode error
            raise PrometheusError("Failed to decode Prometheus response as JSON") from exc

        if payload.get("status") != "success":
            raise PrometheusError(f"Prometheus error: {payload}")

        data = payload.get("data", {})
        result = data.get("result")
        if result is None:
            raise PrometheusError("Prometheus response has no 'data.result' field")

        return result

    def query(self, promql: str) -> List[Dict[str, Any]]:
        """
        Execute an instant Prometheus query (``/api/v1/query``).

        Parameters
        ----------
        promql:
            PromQL expression string to evaluate.

        Returns
        -------
        list of dict
            The ``data.result`` list from the Prometheus response.

        Raises
        ------
        PrometheusError
            If the HTTP request fails or Prometheus returns an error.
        """
        url = f"{self.base_url}/api/v1/query"
        try:
            resp = requests.get(url, params={"query": promql}, timeout=self.timeout)
        except RequestException as exc:
            raise PrometheusError(f"Failed to query Prometheus at {url}: {exc}") from exc
        return self._handle_response(resp)

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute a range Prometheus query (``/api/v1/query_range``).

        Parameters
        ----------
        promql:
            PromQL expression string to evaluate.
        start:
            Start time for the query window (inclusive).
        end:
            End time for the query window (inclusive).
        step:
            Step between samples as a duration string understood by Prometheus,
            for example ``\"60s\"``.

        Returns
        -------
        list of dict
            The ``data.result`` list from the Prometheus response.

        Raises
        ------
        PrometheusError
            If the HTTP request fails or Prometheus returns an error.
        """
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": promql,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": step,
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
        except RequestException as exc:
            raise PrometheusError(f"Failed to query Prometheus at {url}: {exc}") from exc
        return self._handle_response(resp)
