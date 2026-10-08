---
type: Reference
title: Deployment Runbook & Containerization
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Operational instructions for building Docker images and running production services.
tags: [docker, deployment, operations]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Deployment Runbook & Containerization

## Local Development

Start the Pixi-managed FastAPI service:

```bash
pixi run -e dev api
```

Start the local infrastructure stack (MinIO, Prometheus, Grafana):

```bash
docker compose -f docker-compose.local.yaml up -d
```

Start the local MLflow tracking server:

```bash
pixi run -e dev mlflow
```

## Production Deployment (Current)

Deployments to production are gated behind the `taxon-vision-prod` GitHub Environment
and require manual administrator approval before the ARC runner pod executes the
rolling restart of the FastAPI Kubernetes deployment.

## Planned: Jenkins Deploy Stage

Once the Jenkins CI layer is introduced (see
[current_infrastructure.md](current_infrastructure.md)), the deploy step will become
the final stage in the Jenkins pipeline rather than a GitHub Actions step. The Jenkins
`Deploy` stage will run in a CPU-only Kubernetes pod and issue a `kubectl rollout restart`
against the FastAPI deployment, eliminating the need for manual environment approval for
routine model-update deploys while retaining it for infrastructure changes.
