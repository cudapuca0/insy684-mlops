# Build context: repository root (see docker-compose.yml)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/config.yaml

COPY services/simulator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir /app

COPY config.yaml /app/config.yaml
COPY services/simulator /app/service
WORKDIR /app/service

ENTRYPOINT ["python", "simulate.py"]
