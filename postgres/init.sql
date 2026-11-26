-- Runs once, the first time the postgres volume is created.
-- Tables inside `monitoring` are created by the API / monitor (mlops_core/db.py).
CREATE DATABASE mlflow;
CREATE DATABASE monitoring;
