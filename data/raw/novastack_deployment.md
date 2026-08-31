# NovaStack Deployment Runbook

Production services are deployed from immutable container images identified by a Git commit SHA. A release begins in the staging environment, where database compatibility, health checks, and synthetic purchase tests must pass. Production rollout then proceeds region by region rather than globally.

Each region starts with a canary receiving five percent of traffic for ten minutes. The deployment controller automatically rolls back when the five-minute error rate exceeds two percent or p95 latency rises more than 40 percent above the previous version. An operator may also stop a rollout from the release console.

Applications must expose `/health/live` for process liveness and `/health/ready` for dependency readiness. Liveness checks must not contact external systems. Readiness may test critical dependencies but must return within two seconds.

Configuration changes use the same review and rollout process as application code. Secrets come from the managed secret store and must never appear in container images, repository files, or deployment logs. The previous three production images are retained for emergency rollback.

