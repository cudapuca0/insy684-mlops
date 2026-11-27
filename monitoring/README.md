# Monitoring config (Prometheus + Grafana)

Config as code, ready to be mounted by `docker-compose.yml`:

| Service | Mount | Container path |
|---|---|---|
| prometheus | `./monitoring/prometheus/prometheus.yml` | `/etc/prometheus/prometheus.yml` |
| grafana | `./monitoring/grafana/provisioning` | `/etc/grafana/provisioning` |
| grafana | `./monitoring/grafana/dashboards` | `/var/lib/grafana/dashboards` |

Grafana needs `POSTGRES_USER` and `POSTGRES_PASSWORD` in its environment (read by the Postgres datasource).

**Dashboard "Hotel Cancellation Model Monitoring"**
- Prometheus (from the API's `/metrics`): requests/sec, p50/p95/p99 latency, error rate (4xx + 5xx),
  predictions by outcome, share flagged as likely to cancel.
- Postgres `monitoring` database (written by the Evidently monitor): latest drift verdict, share of
  drifted columns, model quality when true outcomes are known, retrains triggered, mean predicted probability.

**Alerts** (folder "MLOps"): data drift (share of drifted columns ≥ 0.5), API error rate > 10%, p95 latency > 1 s.

Metric names match `api/main.py` on `feat/api`: `api_requests_total{endpoint,status}`,
`api_request_latency_seconds{endpoint}`, `model_predictions_total{outcome}`.
Adapted from the `starter-full` reference branch.
