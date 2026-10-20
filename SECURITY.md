# Security policy

## Scope
This is an educational demonstration and is not intended for clinical use.

## Reporting
Do not disclose secrets, patient information or exploitable findings in public issues. Follow your organisation's private vulnerability reporting process.

## Operational controls
- Never log request bodies, patient identifiers, credentials or tokens.
- Only load model artifacts from a trusted pipeline; joblib/pickle deserialization can execute code.
- Keep dependencies pinned, run `pip-audit` and your organisation's Nexus IQ scan on every release candidate.
- Treat Sonar and dependency scans as gates, not proof that the application is vulnerability-free.
- Use non-root containers, read-only mounts/filesystem, least privilege and TLS at ingress.
