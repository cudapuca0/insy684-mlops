FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.17.2 psycopg2-binary==2.9.10

EXPOSE 5000
# Backend store (runs, registry) in Postgres; artifacts proxied by the server
# into the mlflow_artifacts volume, so clients don't need a shared filesystem.
CMD mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/mlflow \
    --artifacts-destination /mlartifacts \
    --serve-artifacts
