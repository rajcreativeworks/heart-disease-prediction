# MLOps Assignment 01 — Heart Disease Prediction

> Replace every placeholder with your own run outputs and screenshots. Do not claim cloud deployment or performance not verified by your own experiments.

## 1. Objective and dataset
- Source: UCI Heart Disease dataset, dataset ID 45.
- Target mapping: original target `num == 0` → 0; `num > 0` → 1.
- Final row count: TODO
- Missing values: TODO

## 2. EDA
Include class balance, missingness, feature distributions and correlation heatmap. Discuss only patterns visible in your outputs.

## 3. Preprocessing and feature engineering
Describe the sklearn pipeline, numeric median imputation/scaling, categorical most-frequent imputation/one-hot encoding, and why transformations are fitted inside CV folds.

## 4. Models and evaluation
Compare logistic regression and random forest using stratified cross-validation. Report holdout accuracy, precision, recall and ROC-AUC; include confusion matrix and ROC curve. Discuss uncertainty due to the small dataset and avoid clinical claims.

## 5. MLflow
Include screenshots of experiment runs, parameters, metrics and artifacts.

## 6. Packaging and API
Document model artifact format, Python/scikit-learn version compatibility, API schema, request ID, `/health`, `/ready`, `/metrics` and `/predict`.

## 7. Quality and security
Include Ruff, pytest/coverage, Bandit, pip-audit, GitHub Actions, SonarQube and Nexus IQ results where available. Record scan date, tool version and any accepted exceptions. A passing local check is not proof of zero vulnerabilities.

## 8. Docker and deployment
Include Docker build/run output, API request/response, container security settings, Kubernetes resources and endpoint verification.

## 9. Monitoring and logging
Explain structured JSON logs, request lifecycle events, error counters, latency and request metrics. Do not log request bodies or patient-identifiable information.

## 10. Limitations and conclusion
This educational model is not a medical device and must not be used for diagnosis, treatment or clinical decisions.
