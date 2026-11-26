# Build from the repository root:  docker build -f api/Dockerfile -t hotel-api .
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY api /app/api
COPY frontend/pipeline/hotel_model.py /app/pipeline/hotel_model.py
COPY model/cancellation_model.joblib /app/model/cancellation_model.joblib
ENV MODEL_PATH=/app/model/cancellation_model.joblib HOTEL_MODEL_DIR=/app/pipeline
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
