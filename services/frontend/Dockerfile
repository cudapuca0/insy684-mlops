# Build context: repository root (see docker-compose.yml)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/config.yaml

COPY services/frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The frontend only needs config loading from mlops_core (no scikit-learn).
COPY src/mlops_core/__init__.py src/mlops_core/config.py /app/mlops_core/
RUN pip install --no-cache-dir pyyaml==6.0.2

COPY config.yaml /app/config.yaml
COPY services/frontend /app/service
WORKDIR /app/service
ENV PYTHONPATH=/app

EXPOSE 5001
CMD ["gunicorn", "-b", "0.0.0.0:5001", "-w", "2", "app:app"]
