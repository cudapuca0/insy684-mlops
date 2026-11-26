import os

import requests
from flask import Flask

app = Flask(__name__)
api_base_url = os.environ["API_BASE_URL"]


@app.get("/")
def home():
    try:
        response = requests.get(f"{api_base_url}/health", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return "<h1>API is unavailable</h1>", 503

    return "<h1>API is reachable</h1>"
