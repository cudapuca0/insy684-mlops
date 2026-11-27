# MLflow tracking server + model registry.
# Backend store: SQLite in the mlflow_data volume by default. Once the Postgres service
# (postgres/init.sql creates an `mlflow` database) is in docker-compose, set
# MLFLOW_BACKEND_STORE_URI=postgresql://<user>:<password>@postgres:5432/mlflow
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=3.1,<4" psycopg2-binary
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
EXPOSE 5000
# --serve-artifacts: clients upload/download models through the server, no shared disk needed.
# --allowed-hosts: accept requests addressed to the compose service name (mlflow:5000).
CMD mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination /mlflow/artifacts --serve-artifacts \
    --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"
