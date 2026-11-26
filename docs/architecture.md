# Architecture

Diagram-as-code (Mermaid). GitHub renders it automatically. Keep it in sync
with `docker-compose.yml` when you change services.

```mermaid
flowchart LR
    %% ---------- Training ----------
    subgraph TRAIN["Training (orchestrated)"]
        DATA[("Dataset<br/>config.yaml → data")]
        FLOW["Prefect flow<br/>training-pipeline"]
        PREFECT["Prefect server<br/>:4200"]
    end

    subgraph TRACK["Tracking & registry"]
        MLFLOW["MLflow server<br/>:5050"]
        REG[("Model Registry<br/>@production")]
    end

    %% ---------- Serving ----------
    subgraph SERVE["Serving"]
        USER(("User"))
        FE["Frontend<br/>Flask :5001"]
        API["Model API<br/>FastAPI :8000"]
    end

    %% ---------- Monitoring ----------
    subgraph MON["Monitoring"]
        SIM["Traffic simulator<br/>(--drift)"]
        PG[("PostgreSQL<br/>predictions + drift_metrics")]
        EV["Evidently monitor"]
        PROM["Prometheus<br/>:9090"]
        GRAF["Grafana<br/>:3000<br/>dashboards + alerts"]
    end

    DATA -->|load + validate| FLOW
    PREFECT -->|schedules / runs| FLOW
    FLOW -->|params, metrics, models| MLFLOW
    FLOW -->|register + set alias| REG
    FLOW -->|reference.csv, test.csv| EV
    MLFLOW --- REG

    USER -->|fills form| FE
    FE -->|POST /predict| API
    API -->|"load model @production"| REG
    API -->|prediction JSON| FE

    SIM -->|POST /predict + /feedback| API
    API -->|log inputs, prediction, label| PG
    PG -->|recent window| EV
    EV -->|drift + quality metrics| PG
    EV -->|"drift: trigger retrain"| PREFECT
    PROM -->|scrape /metrics| API
    GRAF -->|query| PROM
    GRAF -->|query| PG

    %% ---------- CI/CD ----------
    subgraph CICD["GitHub Actions"]
        CI["CI: ruff + pytest<br/>+ API integration test"]
        CD["CD: build images → GHCR"]
    end
    CI -.->|"PR approved + merged to main"| CD
```

## Request path (user → model)

1. The user submits the form in the **frontend** container (`:5001`).
2. The frontend validates the input and sends `POST /predict` to the **API** container (`http://api:8000`).
3. The API validates the payload against `config.yaml` bounds (bad input → HTTP 422), runs the model
   loaded from the **MLflow registry** (`models:/<name>@production`) and returns the label + probability.
4. The API logs the request to **PostgreSQL** and updates its **Prometheus** metrics.
5. The frontend shows a readable result (e.g. "Malignant — 87%").

## Drift → retraining path

1. The **simulator** sends unseen test rows (optionally drifted) and reports the true label via `/feedback`.
2. Every 30 s the **Evidently monitor** compares the latest window of predictions to the training
   reference and writes drift share + accuracy/F1 to `drift_metrics`.
3. **Grafana** shows the change and fires the *Data drift detected* alert.
4. The monitor triggers the Prefect deployment `training-pipeline/retrain`, which retrains on the original
   data plus labelled production data and promotes the new version if it scores at least as well.
5. The API notices the alias moved and hot-reloads the new model version.
