# Kubernetes Anomaly Detection Service

FastAPI-based service that performs anomaly detection on Kubernetes pod metrics
fetched from Prometheus. It uses `StandardScaler` + `IsolationForest` to score
pods based on:

- CPU usage: `rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[1m])`
- Memory usage: `container_memory_usage_bytes{container!="",container!="POD"}`
- Pod restart count: `kube_pod_container_status_restarts_total`

The service exposes:

- REST endpoint `/detect` – anomaly flag and score per pod.
- Prometheus metrics endpoint `/metrics` – Gauges `anomaly_flag` and
  `anomaly_score` per pod.

---

## Project Structure

- `app/`
  - `config.py` – configuration and environment variables.
  - `prometheus/`
    - `api_client.py` – thin Prometheus HTTP API client.
  - `ml/`
    - `anomaly_detector.py` – model training + inference logic.
  - `api/`
    - `main.py` – FastAPI app and HTTP endpoints.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – container image definition.
- `k8s-deployment.yaml` – example Kubernetes Deployment + Service.

---

## Configuration

Runtime configuration lives in `app/config.py` and is driven entirely by
environment variables. You normally do **not** edit `config.py`; instead you set
env vars when running the app (locally, via Docker, or in Kubernetes).

### Config file: `app/config.py`

This module defines the following settings:

- `PROMETHEUS_URL`
  - Default: `http://prometheus:9090`
  - What it is: Base URL of your Prometheus server (`http://host:port`).
  - Required? **Yes** – you must point this at a real Prometheus instance that
    has Kubernetes metrics.

- `NAMESPACE_REGEX`
  - Default: `.*`
  - What it is: Regular expression used to filter `namespace` labels in PromQL
    queries. Only pods whose namespace matches this regex are included.
  - Required? No – but recommended to restrict to namespaces you care about
    (e.g. `default|prod-.*`).

- `POD_LABEL`
  - Default: `pod`
  - What it is: Name of the label in Prometheus metrics that contains the pod
    name (commonly `pod`, sometimes `pod_name` depending on your setup).
  - Required? No – change only if your metrics use a different label key.

- `TRAINING_LOOKBACK_MINUTES`
  - Default: `60`
  - What it is: How many minutes of historical data to use when training the
    IsolationForest model on startup.
  - Required? No – but you need enough history to get a reasonable model.

- `QUERY_STEP_SECONDS`
  - Default: `60`
  - What it is: Step size for Prometheus `/api/v1/query_range` during training
    (i.e. how often to sample within the lookback window).
  - Required? No.

- `REQUEST_TIMEOUT_SECONDS`
  - Default: `5`
  - What it is: HTTP timeout (in seconds) for Prometheus requests.
  - Required? No.

- IsolationForest tuning
  - `IFOREST_N_ESTIMATORS` (default `100`)
    - Number of trees in the IsolationForest.
  - `IFOREST_MAX_SAMPLES` (default `256`)
    - Number of samples to draw per tree.
  - `IFOREST_CONTAMINATION` (default `auto`)
    - Expected proportion of anomalies in the data.
  - Required? No – adjust if you want to tune model behavior.

### Minimal environment variables to set

In most environments you can start with only:

```bash
export PROMETHEUS_URL="http://<your-prom-host>:9090"
export NAMESPACE_REGEX=".*"   # or a narrower regex for your namespaces
```

All other values have sensible defaults and can be tuned later without code
changes.

---

## Running Locally (Python)

Prerequisites:

- Python 3.11+
- Access to a Prometheus instance that scrapes Kubernetes metrics

Install dependencies and run:

```bash
pip install -r requirements.txt

export PROMETHEUS_URL="http://localhost:9090"      # adjust to your Prometheus
export NAMESPACE_REGEX=".*"

uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/healthz`
- Metrics: `http://localhost:8000/metrics`

Note: If Prometheus is unreachable during startup, training will fail but the
service will still come up. `/detect` will return HTTP 503 until training
completes successfully.

---

## Running with Docker

Build the image:

```bash
docker build -t k8s-anomaly-detector .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e PROMETHEUS_URL=http://host.docker.internal:9090 \  # adjust for your setup
  -e NAMESPACE_REGEX='.*' \
  k8s-anomaly-detector
```

Then access:

- `http://localhost:8000/healthz`
- `http://localhost:8000/detect`
- `http://localhost:8000/metrics`

Docker is useful for local development, but **not required** for deploying to
Kubernetes with the provided manifest.

---

## Prometheus Integration

The service exposes metrics on `/metrics` using `prometheus_client`. It provides:

```text
anomaly_flag{pod="nginx-xx"} 0
anomaly_score{pod="nginx-xx"} -0.12
```

### Local Prometheus (static scrape)

In `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'k8s-anomaly-detector'
    static_configs:
      - targets:
          - 'localhost:8000'  # or host.docker.internal:8000
    metrics_path: /metrics
```

Reload Prometheus and you should see job `k8s-anomaly-detector` and the metrics.

### Kubernetes Prometheus (kubernetes_sd_configs)

Assuming you deployed `k8s-deployment.yaml` into the `default` namespace and the
service name is `k8s-anomaly-detector`:

```yaml
scrape_configs:
  - job_name: 'k8s-anomaly-detector'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - default
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: k8s-anomaly-detector
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        action: keep
        regex: http
    metrics_path: /metrics
```

If you are using Prometheus Operator, create a `ServiceMonitor` targeting the
`k8s-anomaly-detector` service instead, with `path: /metrics`.

---

## Kubernetes Deployment

You can deploy to Kubernetes **without** building or hosting your own image by
using the stock `python:3.11-slim` image and mounting this project as a
`ConfigMap`.

### 1. Create the ConfigMap with your code

From the project root (where `app/` and `requirements.txt` live):

```bash
rm -f code.tar.gz
tar czf code.tar.gz app requirements.txt

kubectl create configmap anomaly-detector-code --from-file=code.tar.gz
```

This will create a `ConfigMap` named `anomaly-detector-code` that contains the
source code and `requirements.txt` packaged as a single tarball. The deployment
mounts it at `/config` and unpacks it into `/workspace` at container startup.

### 2. Deploy the manifest

The file `k8s-deployment.yaml` is already configured to:

- Use `python:3.11-slim` as the base image.
- Mount the `anomaly-detector-code` ConfigMap at `/config`.
- Run:
  ```bash
  mkdir -p /workspace &&
  tar xzf /config/code.tar.gz -C /workspace &&
  cd /workspace &&
  pip install --no-cache-dir -r requirements.txt &&
  uvicorn app.api.main:app --host 0.0.0.0 --port 8000
  ```

Apply it:

```bash
kubectl apply -f k8s-deployment.yaml
```

Key points:

- Deployment runs container exposing port 8000 and probes `/healthz`.
- Service exposes the app inside the cluster as `k8s-anomaly-detector:8000`.
- Environment variables `PROMETHEUS_URL` and `NAMESPACE_REGEX` are configurable
  directly in the manifest.

### 3. Verify the pod and service

Check that the pod is running:

```bash
kubectl get pods -l app=k8s-anomaly-detector
kubectl logs -f deploy/k8s-anomaly-detector
```

Port-forward for local testing:

```bash
kubectl port-forward svc/k8s-anomaly-detector 8000:8000
```

Then open:

- `http://localhost:8000/healthz`
- `http://localhost:8000/detect`
- `http://localhost:8000/metrics`

### 4. Update or delete the deployment

- To update code or dependencies:
  1. Recreate the ConfigMap with the latest files:
     ```bash
     kubectl delete configmap anomaly-detector-code
     rm -f code.tar.gz
     tar czf code.tar.gz app requirements.txt
     kubectl create configmap anomaly-detector-code --from-file=code.tar.gz
     ```
  2. Re-apply the manifest:
     ```bash
     kubectl apply -f k8s-deployment.yaml
     ```

- To delete the deployment and service:
  ```bash
  kubectl delete -f k8s-deployment.yaml
  ```

- (Optional) To also remove the ConfigMap:
  ```bash
  kubectl delete configmap anomaly-detector-code
  ```

---

## API Overview

- `GET /healthz`
  - Returns service status, whether the model is trained, and configured
    `PROMETHEUS_URL`.

- `GET /detect`
  - On-demand anomaly detection over current metrics.
  - Response:
    ```json
    {
      "pods": [
        {
          "pod": "nginx-xxx",
          "cpu": 0.0012,
          "memory": 123456789.0,
          "restarts": 0.0,
          "anomaly_flag": 0,
          "anomaly_score": -0.12
        }
      ]
    }
    ```

- `GET /metrics`
  - Prometheus metrics endpoint exposing `anomaly_flag` and `anomaly_score`
    per pod.

---

## Notes

- The quality of anomaly detection depends on:
  - Having enough historical data in Prometheus (default 60 minutes).
  - Correct Prometheus queries and label conventions (especially pod label).
  - Appropriate IsolationForest configuration (`IFOREST_CONTAMINATION`, etc.).

Feel free to adjust the model configuration and PromQL queries in
`app/ml/anomaly_detector.py` and `app/config.py` to better fit your cluster and
workload characteristics.
