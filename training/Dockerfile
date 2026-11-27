# Build from the repository root:  docker build -f training/Dockerfile -t hotel-training .
# Default command registers the committed model if the registry is empty (no CSV needed).
# Full training:  docker compose run --rm model-registrar --data /data/hotel_bookings.csv
FROM python:3.11-slim
WORKDIR /app
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY frontend/pipeline/hotel_model.py frontend/pipeline/train.py /app/frontend/pipeline/
COPY training /app/training
COPY model/cancellation_model.joblib /app/model/cancellation_model.joblib
COPY tests/data/hotel_bookings_sample.csv /app/tests/data/hotel_bookings_sample.csv
ENTRYPOINT ["python", "training/train_mlflow.py"]
CMD ["--register-joblib", "model/cancellation_model.joblib", "--if-empty"]
