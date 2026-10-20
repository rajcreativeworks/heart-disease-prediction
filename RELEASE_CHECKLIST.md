# Release checklist — required before publishing/deployment

This repository includes security and quality gates, but it has not been scanned by your
SonarQube/SonarCloud instance or Sonatype Nexus IQ policy server. Do not describe it as
"zero vulnerabilities" until the actual target environment has completed those scans.

- [ ] Create a fresh Python 3.12 environment and install requirements.
- [ ] Run `ruff check .`.
- [ ] Run `bandit -q -r src`.
- [ ] Run `pytest --cov=heart_ml --cov-report=term-missing --cov-report=xml:coverage.xml`.
- [ ] Run `pip-audit -r requirements.txt`; review and remediate every applicable finding.
- [ ] Run the Sonar scan against this commit and meet the configured quality gate.
- [ ] Run Nexus IQ policy evaluation against the same source/artifact/container image intended for release.
- [ ] Build the Docker image and scan the image with the organisation-approved container scanner.
- [ ] Verify runtime health/readiness and the prediction endpoint.
- [ ] Record scan date, scanner/database version, findings and approved exceptions.
- [ ] Verify no credentials, data files, model artifacts or logs are staged for Git.
