# Heart Disease Prediction educational template

A reproducible MLOps assignment repository covering UCI data acquisition, EDA, preprocessing/model selection, MLflow tracking, API packaging, tests, Docker, Kubernetes, CI, and security scanning.

> **Safety and scope:** This is an educational project, not a medical device. Do not use predictions for diagnosis or treatment. The model is trained on a small historical dataset and is not clinically validated.

## Repository contents

- `src/heart_ml/data.py` — UCI acquisition and schema validation
- `src/heart_ml/eda.py` — reproducible EDA charts and CSVs
- `src/heart_ml/train.py` — preprocessing pipelines, two classifiers, stratified CV, tuning, holdout metrics, plots and MLflow
- `src/heart_ml/api.py` — FastAPI endpoints, request IDs, structured logs and Prometheus metrics
- `tests/` — API, schema and model tests
- `.github/workflows/ci.yml` — lint, Bandit, dependency audit, tests and coverage
- `sonar-project.properties` — SonarQube project configuration
- `k8s/` — example Kubernetes deployment and service manifests

## Prerequisites

- Python 3.12.x
- Docker (optional, for container execution)
- Git and a GitHub repository
- Internet access for initial dataset acquisition and dependency installation

## Local setup

Linux/macOS:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Windows PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

## Run the workflow

```bash
python -m heart_ml.data --output data/raw/heart_disease.csv
python -m heart_ml.eda --input data/raw/heart_disease.csv --output reports/eda
python -m heart_ml.train --data data/raw/heart_disease.csv --output models/model.joblib --mlflow
```

Training logs JSON-formatted entry/exit and error events to stdout. MLflow tracking uses the local `mlruns/` directory by default. The dataset, model, MLflow runs and generated reports are ignored by Git; store required assignment evidence separately and only commit small, non-sensitive artifacts when appropriate.

## API

Start after training:
```bash
uvicorn heart_ml.api:app --host 0.0.0.0 --port 8000 --no-access-log
```

- `GET /health` — liveness
- `GET /ready` — readiness (503 if model not loaded)
- `GET /metrics` — Prometheus metrics
- `POST /predict` — prediction and probabilities

Example request:
```bash
curl -sS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":58,"sex":1,"cp":4,"trestbps":160,"chol":300,"fbs":1,"restecg":2,"thalach":110,"exang":1,"oldpeak":4.0,"slope":3,"ca":2,"thal":7}'
```

Logs include application lifecycle, request entry/exit, duration, request ID, validation failures and exceptions. They intentionally **do not** log input payloads or feature values. Set `LOG_LEVEL=DEBUG` for diagnostic events; keep `INFO` in normal operation. Avoid DEBUG in production unless temporary and approved.

## Tests and security checks

```bash
ruff check .
bandit -q -r src
pytest --cov=heart_ml --cov-report=term-missing --cov-report=xml:coverage.xml
pip-audit -r requirements.txt
```

`pip-audit` queries its vulnerability database at run time. GitHub Actions fails when the dependency audit fails; do not suppress findings blindly. The requirements file pins direct dependencies, but transitive dependencies and vulnerability advisories can change.

### SonarQube / SonarCloud
The `sonar-project.properties` file provides basic source/test configuration. Configure your Sonar project and CI integration with your organisation's Sonar token stored as a GitHub Actions secret. A config file does not guarantee a clean quality gate; run the scan and fix the findings. Sonar rules and quality gates vary by organisation.

### Sonatype Nexus IQ
Nexus IQ policy evaluation must run in your organisation's Nexus IQ Server/CLI or configured repository manager. This repository cannot guarantee zero Nexus vulnerabilities without access to that server and its current policy/advisory data. Scan the exact commit/image that you intend to release; resolve critical/high policy violations and document approved exceptions. Do not commit credentials or API tokens.

## Docker

Train the model first (the `models/` folder is intentionally not committed):
```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  -e MODEL_PATH=/app/models/model.joblib \
  -e LOG_LEVEL=INFO \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  heart-disease-api:local
```

The image runs as a non-root user and uses a health check. Joblib uses Python pickle-based serialization; only load artifacts created by a trusted pipeline, and keep Python/scikit-learn versions compatible. Do not load untrusted model files.

Compose is also available:
```bash
docker compose up --build
```

## Kubernetes

The sample manifest is a local-cluster starting point. Before applying it:
1. Build or push the image to a registry your cluster can pull from.
2. Update `k8s/deployment.yaml` image.
3. Replace the sample `hostPath` with an approved PersistentVolume, read-only mounted model volume, or a secure model-fetch/init-container design for your cluster.
4. Train the model and ensure the artifact exists.
5. Review security, resource limits, ingress/TLS, secrets, monitoring and storage for your environment.

```bash
kubectl apply -f k8s/
kubectl rollout status deployment/heart-disease-api
kubectl port-forward service/heart-disease-api 8000:80
```

## GitHub first push

Create an **empty** GitHub repository (do not initialize it with a README or license), then from this directory:
```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial MLOps project with CI and security checks"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
Review `git status` before committing. Never commit `.env`, access tokens, raw data, generated model files, logs or MLflow artifact stores.

## Important reproducibility note

The model artifact is deliberately excluded from Git. Re-create it with the training command in a trusted environment, and keep the Python 3.12 / scikit-learn 1.6.1 environment consistent between training and serving. Pinning the direct dependencies helps reproducibility but is not a substitute for a full lock file and repeatable vulnerability scans.
