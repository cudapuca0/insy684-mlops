FROM python:3.11-slim
WORKDIR /app
COPY simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulator/ ./simulator/
# The dataset is mounted at runtime (docker compose: ./data:/app/data) and is not baked into the image.
ENV DATA_PATH=/app/data/raw/hotel_bookings.csv \
    API_URL=http://api:8000
ENTRYPOINT ["python", "simulator/simulate.py"]
