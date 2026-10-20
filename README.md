# Heart Disease Prediction --- End-to-End MLOps on GKE

An end-to-end machine-learning operations (MLOps) project that trains
and evaluates heart-disease classifiers, packages the prediction API
into a Docker image, deploys it to Google Kubernetes Engine (GKE), and
monitors the running API with Prometheus and Grafana.

> **Project status:** The application and monitoring stack have been
> deployed to GKE, and the `Deploy Monitoring Stack` GitHub Actions
> workflow is working. The Grafana dashboard is provisioned, but its
> panels still require final query/metric validation and may show **No
> data**. Do not treat dashboard visualizations as verified until they
> display metrics.

## Contents

-   [Overview](#overview)
-   [Architecture](#architecture)
-   [Technology stack](#technology-stack)
-   [Repository layout](#repository-layout)
-   [Prerequisites](#prerequisites)
-   [1. Clone and set up locally](#1-clone-and-set-up-locally)
-   [2. Run quality checks and tests](#2-run-quality-checks-and-tests)
-   [3. Prepare data and train the
    model](#3-prepare-data-and-train-the-model)
-   [4. Run the API locally](#4-run-the-api-locally)
-   [5. Build and run the Docker
    image](#5-build-and-run-the-docker-image)
-   [6. CI/CD with GitHub Actions](#6-cicd-with-github-actions)
-   [7. Deploy to GKE](#7-deploy-to-gke)
-   [8. Deploy Prometheus and Grafana](#8-deploy-prometheus-and-grafana)
-   [9. Verify the deployment](#9-verify-the-deployment)
-   [10. Capture submission evidence](#10-capture-submission-evidence)
-   [Configuration and secrets](#configuration-and-secrets)
-   [Troubleshooting](#troubleshooting)
-   [Limitations and next steps](#limitations-and-next-steps)

## Overview

The project demonstrates the main stages of an MLOps lifecycle:

1.  Prepare the dataset and train candidate models.
2.  Compare Logistic Regression and Random Forest models using
    cross-validation and evaluation metrics.
3.  Save the selected model and its metadata/metrics as artifacts.
4.  Run automated code-quality, test, security and container checks
    through GitHub Actions.
5.  Publish a versioned Docker image to GitHub Container Registry
    (GHCR).
6.  Deploy the FastAPI inference service to GKE Autopilot.
7.  Scrape API and Kubernetes metrics with Prometheus.
8.  Provision a Grafana dashboard for operational monitoring.

The model training run previously reported approximately **88.5% test
accuracy** and **0.965 ROC-AUC**. These are historical run results, not
a guarantee that every future training run will produce identical
values. Use the metrics artifacts from the latest successful run as the
authoritative results for submission.

## Architecture

The Mermaid diagram below is source-controlled and rendered
automatically by GitHub. It shows the training and deployment path as
well as the runtime monitoring path.

``` mermaid
flowchart TB
    subgraph Dev["Source control and CI/CD"]
        Git["GitHub repository"]
        CI["GitHub Actions CI<br/>quality • tests • security"]
        Train["Data preparation and model training<br/>Logistic Regression + Random Forest"]
        Artifacts["Training artifacts<br/>model.joblib • metadata • metrics"]
        Publish["Publish image workflow"]
        GHCR["GitHub Container Registry<br/>ghcr.io/rajcreativeworks/heart-disease-prediction"]
        Git --> CI
        CI --> Train
        Train --> Artifacts
        Artifacts --> Publish
        Publish --> GHCR
    end

    subgraph GKE["Google Kubernetes Engine Autopilot"]
        Deploy["GitHub Actions deployment"]
        API["Heart Disease FastAPI<br/>/predict • /health • /ready • /metrics"]
        Service["Kubernetes ClusterIP Service"]
        Deploy --> API
        Service --> API
    end

    GHCR --> Deploy
    Client["Client / API consumer"] --> Service

    subgraph Monitoring["Monitoring namespace"]
        Prom["Prometheus"]
        Graf["Grafana"]
        Dashboard["Heart Disease API dashboard"]
        Prom --> Graf
        Graf --> Dashboard
    end

    API -- "scrapes /metrics" --> Prom
```

### Runtime request flow

1.  A client sends a request to the Kubernetes Service.
2.  The Service routes the request to the FastAPI pod.
3.  The API loads the saved model artifact and returns a prediction.
4.  Prometheus periodically scrapes the API's `/metrics` endpoint.
5.  Grafana queries Prometheus to visualize the collected metrics.

### Training and release flow

1.  A change is pushed to GitHub.
2.  GitHub Actions runs the CI workflow.
3.  The training pipeline prepares data, evaluates candidate models and
    writes artifacts.
4.  The image-publishing workflow builds and pushes the image to GHCR
    using the training artifacts.
5.  The deployment workflow deploys the application to GKE.
6.  The monitoring workflow installs/upgrades Prometheus and Grafana and
    provisions the dashboard.

## Technology stack

  -----------------------------------------------------------------------
  Area                                Tools
  ----------------------------------- -----------------------------------
  Language and ML                     Python, pandas, scikit-learn,
                                      joblib

  API                                 FastAPI, Uvicorn

  Experiment tracking                 MLflow

  Testing and code quality            Project-configured tests, linting
                                      and security checks

  Containerization                    Docker

  Source and CI/CD                    GitHub, GitHub Actions

  Image registry                      GitHub Container Registry (GHCR)

  Kubernetes                          Google Kubernetes Engine (GKE)
                                      Autopilot, Kubernetes manifests,
                                      Helm

  Metrics                             Prometheus

  Dashboards                          Grafana
  -----------------------------------------------------------------------

## Repository layout

The layout below highlights the important files used by the pipeline.
Some additional project files may exist.

``` text
heart-disease-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── publish-image.yml
│       ├── deploy.yml
│       └── deploy-monitoring.yml
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   ├── prometheus/
│   │   └── values.yaml
│   └── grafana/
│       ├── values.yaml
│       └── dashboards/
│           └── heart-disease-api.json
├── src/
│   └── heart_ml/
│       ├── api.py
│       └── ... (data, training and model modules)
├── tests/
├── models/                 # generated model artifacts; may be created by training/CI
├── reports/                # generated evaluation reports
├── Dockerfile
├── pyproject.toml
└── README.md
```

Confirm the actual workflow filenames in `.github/workflows/`; if yours
differ, use the names that exist in your repository.

## Prerequisites

### Local development

-   Git
-   Python 3.12 (the development environment used for this project was
    Python 3.12.10)
-   pip and `venv`
-   Docker Desktop (optional for local container testing)

### Cloud deployment

-   A Google Cloud project with GKE enabled
-   `gcloud`, `kubectl` and Helm available in the deployment environment
-   A GKE cluster and a Google service account for GitHub Actions
    authentication
-   GitHub Actions secrets and variables configured as described below
-   Permission to push images to GHCR

## 1. Clone and set up locally

Run in PowerShell on Windows:

``` powershell
git clone https://github.com/rajcreativeworks/heart-disease-prediction.git
cd heart-disease-prediction

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If the project does not define a `dev` optional dependency group in
`pyproject.toml`, install the dependencies using the method defined by
the repository (for example, `pip install -r requirements.txt` if that
file is present). Do not install both methods blindly; use the
dependency file maintained by the project.

Verify the Python environment:

``` powershell
python --version
python -m pip --version
```

Expected Python version: `3.12.x`.

## 2. Run quality checks and tests

Use the commands configured in `pyproject.toml` and
`.github/workflows/ci.yml`. A typical local test command is:

``` powershell
python -m pytest -v
```

If pytest coverage is configured:

``` powershell
python -m pytest --cov=src/heart_ml --cov-report=term-missing
```

Run the exact lint, formatting and security commands from the CI
workflow before submitting. The GitHub Actions run is the source of
truth for whether the full CI pipeline passed.

## 3. Prepare data and train the model

The project training pipeline compares Logistic Regression and Random
Forest and selects a model based on its evaluation procedure.

Run the data-preparation and training commands specified in
`.github/workflows/ci.yml` or the package's documented CLI. To inspect
the configured commands locally:

``` powershell
Get-Content .github\workflows\ci.yml
Get-ChildItem src\heart_ml -Recurse
```

After training, confirm that the expected model and reports were
produced. The deployment image expects the model artifacts in the
`models/` directory (including `models/model.joblib`).

Useful checks:

``` powershell
Get-ChildItem models
Get-ChildItem reports -Recurse
```

MLflow tracking is used for experiment runs. If the project environment
is configured to use a local MLflow file store, launch its UI with:

``` powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`. Use the configured MLflow tracking
URI if it differs from the local default.

> Training commands can change as the pipeline evolves. Prefer the
> command already used by the current CI workflow instead of copying a
> stale command into a submission runbook.

## 4. Run the API locally

Activate the virtual environment and ensure the trained model exists at
the path expected by the API. From the repository root, the package can
commonly be launched as:

``` powershell
python -m uvicorn heart_ml.api:app --app-dir src --host 127.0.0.1 --port 8000
```

If the application object or startup command in `src/heart_ml/api.py`
has a different name, use the exact import target defined in that file
and in the Dockerfile.

Useful endpoints:

  Endpoint          Purpose
  ----------------- ----------------------------------
  `GET /health`     Liveness/health check
  `GET /ready`      Readiness and model availability
  `GET /metrics`    Prometheus metrics
  `POST /predict`   Generate a prediction

OpenAPI documentation is usually available at
`http://127.0.0.1:8000/docs` when enabled by FastAPI.

Check health and readiness:

``` powershell
curl.exe -i http://127.0.0.1:8000/health
curl.exe -i http://127.0.0.1:8000/ready
curl.exe -i http://127.0.0.1:8000/metrics
```

For `/predict`, send a JSON body that matches the request schema defined
in the API code. Do not use arbitrary clinical values or rely on a
sample schema that differs from the implementation.

## 5. Build and run the Docker image

Run these commands from the repository root. Training artifacts must
exist in `models/` because the Docker build copies that directory into
the image.

``` powershell
docker build -t heart-disease-prediction:local .
docker run --rm -p 8000:8000 heart-disease-prediction:local
```

In another terminal:

``` powershell
curl.exe -i http://127.0.0.1:8000/health
curl.exe -i http://127.0.0.1:8000/ready
```

Stop the container with `Ctrl+C` in the terminal where it is running.

The production image is published to:

``` text
ghcr.io/rajcreativeworks/heart-disease-prediction:latest
```

The publishing workflow also creates a commit-specific tag. Prefer
immutable commit tags for reproducible releases where supported by your
deployment workflow.

## 6. CI/CD with GitHub Actions

The repository uses GitHub Actions to automate quality checks, tests,
model training/artifact generation, image publishing, application
deployment and monitoring deployment.

Review the workflows under `.github/workflows/`. Typical
responsibilities are:

-   **CI:** quality checks, tests, security checks, container
    validation, data preparation and model training.
-   **Image publishing:** retrieve training artifacts and build/push the
    Docker image to GHCR.
-   **Application deployment:** authenticate to Google Cloud and deploy
    the API to GKE.
-   **Monitoring deployment:** install/upgrade Prometheus and Grafana
    and provision the Grafana dashboard.

### Run and inspect workflows

1.  Push committed changes to the configured branch, or use **Actions →
    Run workflow** if the workflow supports manual dispatch.
2.  Open the run in the GitHub Actions tab.
3.  Verify every required job succeeds.
4.  If a workflow is triggered by another workflow, verify the
    downstream workflow completed successfully too.

## 7. Deploy to GKE

The current deployment environment uses:

-   GCP project: `heart-risk-mlops`
-   GKE cluster: `heart-disease-cluster`
-   Region: `asia-south1`
-   Application namespace: `default`
-   Monitoring namespace: `monitoring`

Authenticate with an account that has the required permissions:

``` bash
gcloud auth login
gcloud config set project heart-risk-mlops
gcloud container clusters get-credentials heart-disease-cluster \
  --region asia-south1 \
  --project heart-risk-mlops
```

Verify the active context:

``` bash
kubectl config current-context
kubectl get nodes
```

Inspect application resources:

``` bash
kubectl get deployments,pods,services -n default
kubectl describe deployment heart-disease-api -n default
```

The preferred deployment path is the version-controlled GitHub Actions
deployment workflow. If you need to apply manifests manually for
diagnosis, review them first and ensure the image tag and model artifact
are correct:

``` bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api -n default
```

Use the workflow for the official deployment when it is responsible for
updating the image and manifests.

### Test the deployed API from Cloud Shell

Start port forwarding:

``` bash
kubectl port-forward -n default service/heart-disease-api 8000:80
```

In a second terminal:

``` bash
curl -i http://localhost:8000/health
curl -i http://localhost:8000/ready
curl -i http://localhost:8000/metrics
```

Use a schema-valid JSON payload to test `POST /predict`.

## 8. Deploy Prometheus and Grafana

The monitoring stack is managed through the `Deploy Monitoring Stack`
GitHub Actions workflow and version-controlled Helm values.

Relevant files:

``` text
monitoring/prometheus/values.yaml
monitoring/grafana/values.yaml
monitoring/grafana/dashboards/heart-disease-api.json
```

### Run the monitoring workflow

1.  Open the GitHub repository.
2.  Select **Actions → Deploy Monitoring Stack**.
3.  Run it on the intended branch if manual dispatch is enabled.
4.  Confirm that authentication, GKE credentials, Prometheus Helm
    upgrade, Grafana Helm upgrade, dashboard provisioning and
    verification steps succeed.

### Inspect the monitoring stack

``` bash
kubectl get deployments,pods,services -n monitoring
kubectl get configmaps -n monitoring
```

### Access Prometheus through port forwarding

``` bash
kubectl port-forward -n monitoring service/prometheus-server 9090:80
```

In a second terminal, check:

``` bash
curl -s http://localhost:9090/-/healthy
curl -s http://localhost:9090/-/ready
```

Query the application target:

``` bash
curl -sG 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=up{job="heart-disease-api"}'
```

A returned value of `1` indicates the target is up.

Inspect all targets:

``` bash
curl -s 'http://localhost:9090/api/v1/targets'
```

### Access Grafana through port forwarding

``` bash
kubectl port-forward -n monitoring service/grafana 3000:80
```

Open `http://localhost:3000`. Sign in with the administrator username
and password configured by the deployment. The password should be
supplied through the configured Kubernetes Secret/GitHub Actions secret;
do not store it in this README.

Open **Dashboards → Heart Disease API Monitoring** (or the dashboard
title currently provisioned in your environment).

The dashboard is expected to visualize request rate, p95 latency,
prediction counts, API errors and target health. Confirm the panels
display actual data before claiming the Grafana visualization is
complete. A target being `UP` alone does not prove every dashboard query
is correct.

## 9. Verify the deployment

Run the following checks before submission:

``` bash
# Application resources
kubectl get deployments,pods,services -n default

# Monitoring resources
kubectl get deployments,pods,services -n monitoring

# Recent application logs
kubectl logs deployment/heart-disease-api -n default --tail=100

# Recent Prometheus logs
kubectl logs deployment/prometheus-server -n monitoring \
  -c prometheus-server --tail=100

# Prometheus health and readiness (while port-forward is active)
curl -s http://localhost:9090/-/healthy
curl -s http://localhost:9090/-/ready
```

For each deployment, confirm that the available replica count matches
the desired replica count. Review pod restart counts and events if a pod
is not healthy.

## 10. Capture submission evidence

Capture and save screenshots or text output for:

1.  Successful CI workflow.
2.  Successful image-publishing workflow and GHCR package/tag.
3.  Successful application deployment workflow.
4.  Successful **Deploy Monitoring Stack** workflow.
5.  Application and monitoring Kubernetes pods/services.
6.  API health/readiness and one valid prediction response.
7.  Prometheus health/readiness and the `heart-disease-api` target
    status.
8.  Grafana dashboard showing populated panels once query
    troubleshooting is complete.
9.  Repository tree and key version-controlled configuration files.

Do not expose credentials, tokens, secret values or sensitive patient
data in screenshots.

## Configuration and secrets

The deployment workflow uses GitHub Actions secrets/variables for Google
Cloud authentication and the Grafana administrator password. Names
currently used in this project include:

-   `GCP_WORKLOAD_IDENTITY_PROVIDER` --- GitHub Actions Workload
    Identity Provider resource.
-   `GCP_SERVICE_ACCOUNT` --- Google service account used by GitHub
    Actions.
-   `GRAFANA_ADMIN_PASSWORD` --- secret used to provision the Grafana
    administrator password.

Check the workflow YAML for the exact variable/secret names and the
required permissions. Prefer Workload Identity Federation over
long-lived service-account key files. Grant only the IAM and Kubernetes
permissions needed by the workflows.

Never commit secret values into YAML, shell scripts, Docker images or
this README.

## Troubleshooting

### API pod does not start

``` bash
kubectl describe pod -n default <pod-name>
kubectl logs -n default <pod-name> --all-containers=true --tail=200
```

Check image access, environment variables, model-file paths and
readiness/liveness probe configuration.

### Prometheus is healthy but API target is down

``` bash
kubectl get service heart-disease-api -n default
kubectl get endpoints heart-disease-api -n default
kubectl logs deployment/prometheus-server -n monitoring \
  -c prometheus-server --tail=100
```

Check the scrape target hostname/port, API `/metrics` endpoint and
Service endpoints.

### Grafana dashboard shows "No data"

1.  Confirm the Prometheus datasource URL is
    `http://prometheus-server.monitoring.svc.cluster.local`.
2.  Test the datasource connection from Grafana, if the UI exposes the
    action.
3.  Query the metric names directly in Prometheus.
4.  Generate API traffic and wait for at least one scrape interval.
5.  Inspect the PromQL query in each affected dashboard panel.
6.  Check whether the metric names and labels used in the dashboard
    match those emitted by the API.

Do not change dashboard queries until the metric names and labels have
been confirmed from `/metrics` or Prometheus.

### Monitoring workflow fails

Review the first error in the failed GitHub Actions step. Common areas
to check are Workload Identity Federation configuration, GKE
credentials, IAM/Kubernetes permissions, Helm values, and whether a
cluster-scoped resource requires permissions the deployer does not have.

## Limitations and next steps

-   Finish validating the Grafana panel queries against the live metric
    names.
-   Add alerting rules for API availability, elevated error rates and
    high latency if required by the assignment.
-   Consider persistence and retention requirements before using the
    monitoring stack for long-running production workloads.
-   Review the deployer's IAM permissions and replace broad permissions
    with least-privilege access where possible.
-   Document the exact training command, prediction request schema and
    final metrics from the latest successful run.

## Disclaimer

This project is an educational machine-learning and MLOps demonstration.
Model predictions are not a medical diagnosis and must not be used as a
substitute for professional medical advice.
