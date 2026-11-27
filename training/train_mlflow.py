"""Train the hotel cancellation model with MLflow tracking and register it.

Two modes:

  Full training (needs the CSV):
      python training/train_mlflow.py --data frontend/data/raw/hotel_bookings.csv

    Logs one parent run with a nested run per candidate, all fit on the training period
    and compared on the validation period:
      - baseline          majority class (naive reference)
      - extra_trees_v2    Extra Trees, feature version V2
      - mlp_v0            MLP, feature version V0
      - ensemble          equal-weight average of the two (the final recipe, threshold 0.45)
    then a final run that refits the ensemble on train + validation (as train.py does),
    evaluates it once on the test period, registers it as `hotel-cancellation` and moves
    the `production` alias to it if it scores at least as well as the current production
    model on the same test period.

  Register the committed model (no CSV needed, used by docker compose on first start):
      python training/train_mlflow.py --register-joblib model/cancellation_model.joblib --if-empty

The tracking server comes from MLFLOW_TRACKING_URI (for example http://mlflow:5000).
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(os.environ.get("HOTEL_MODEL_DIR", ROOT / "frontend" / "pipeline"))
sys.path.insert(0, str(PIPELINE_DIR))

from hotel_model import (  # noqa: E402
    DECISION_THRESHOLD,
    CancellationEnsemble,
    make_extra_trees_v2,
    make_mlp_v0,
)
from train import load_and_split  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "hotel-cancellation")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "hotel-cancellation")
SELECTION_METRIC = "roc_auc"  # used to compare a new final model with the current production one


# ---------------------------------------------------------------------------
# Plain functions (no MLflow) so they can be unit tested
# ---------------------------------------------------------------------------
def evaluate(y_true, proba, threshold=DECISION_THRESHOLD):
    """Same metrics as train.py, plus the predicted cancel rate."""
    pred = (proba > threshold).astype(int)
    metrics = {
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "predicted_cancel_rate": float(pred.mean()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    return {k: float(v) for k, v in metrics.items()}


def candidates(X_train):
    """The candidates compared on the validation period: name -> (params, unfitted model)."""
    return {
        "extra_trees_v2": (
            {"model": "ExtraTrees", "features": "V2", "n_estimators": 90, "max_depth": 18,
             "min_samples_leaf": 4},
            make_extra_trees_v2(X_train),
        ),
        "mlp_v0": (
            {"model": "MLP", "features": "V0", "hidden_layer_sizes": "(48,)", "max_iter": 90},
            make_mlp_v0(X_train),
        ),
        "ensemble": (
            {"model": "ExtraTrees V2 + MLP V0 (equal weight)", "threshold": DECISION_THRESHOLD},
            CancellationEnsemble(members={
                "Extra Trees V2": make_extra_trees_v2(X_train),
                "MLP V0": make_mlp_v0(X_train),
            }),
        ),
    }


def baseline_proba(y_train, n_rows):
    """Naive reference: every booking gets the training cancellation rate."""
    return np.full(n_rows, float(np.mean(y_train)))


def final_model(X_train, X_dev, y_dev):
    """The final recipe from train.py: ensemble refit on train + validation."""
    model = CancellationEnsemble(members={
        "Extra Trees V2": make_extra_trees_v2(X_train),
        "MLP V0": make_mlp_v0(X_train),
    })
    return model.fit(X_dev, y_dev)


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------
def log_model(model, X_example):
    """Log the model with hotel_model.py packaged next to it, so it loads anywhere."""
    import mlflow
    from mlflow.models import infer_signature

    example = X_example.head(3)
    return mlflow.sklearn.log_model(
        model,
        name="model",
        code_paths=[str(PIPELINE_DIR / "hotel_model.py")],
        # The ensemble uses our own classes (hotel_model.py), which MLflow's default skops
        # format refuses to load; cloudpickle stores them like joblib does.
        serialization_format="cloudpickle",
        signature=infer_signature(example, model.predict_proba(example)),
        input_example=example,
    )


def current_production(client):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception:
        return None


def register_and_promote(model_uri, test_score, X_test, y_test, force=False):
    """Register the run's model; move the alias if it is at least as good as production."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = mlflow.register_model(model_uri, MODEL_NAME).version

    current = current_production(client)
    current_score = None
    if current is not None and X_test is not None:
        current_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        current_score = evaluate(y_test, current_model.predict_proba(X_test)[:, 1]).get(SELECTION_METRIC)

    promote = force or current is None or current_score is None or (
        test_score is not None and test_score >= current_score
    )
    if promote:
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
        print(f"Registered {MODEL_NAME} v{version} and set @{MODEL_ALIAS} "
              f"(test {SELECTION_METRIC}={test_score}, previous={current_score})")
    else:
        print(f"Registered {MODEL_NAME} v{version}; kept @{MODEL_ALIAS} on v{current.version} "
              f"(test {SELECTION_METRIC} {current_score:.4f} > {test_score:.4f})")
    return version, promote


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def train_and_register(data_path, force_promote=False):
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_split(data_path)

    with mlflow.start_run(run_name="training-pipeline") as parent:
        mlflow.log_params({
            "data": Path(data_path).name, "split": "chronological 70/15/15 by arrival date",
            "n_train": len(X_train), "n_valid": len(X_valid), "n_test": len(X_test),
        })

        # 1. Candidates: fit on train, compare on validation.
        with mlflow.start_run(run_name="baseline", nested=True):
            mlflow.log_param("model", "majority-class rate")
            mlflow.log_metrics({f"val_{k}": v for k, v in
                                evaluate(y_valid, baseline_proba(y_train, len(y_valid))).items()})

        for name, (params, model) in candidates(X_train).items():
            print(f"Fitting {name} on the training period...")
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(params)
                model.fit(X_train, y_train)
                scores = evaluate(y_valid, model.predict_proba(X_valid)[:, 1])
                mlflow.log_metrics({f"val_{k}": v for k, v in scores.items()})
                print(f"  validation: {scores}")

        # 2. Final model: refit on train + validation, evaluate once on test, register.
        X_dev = pd.concat([X_train, X_valid], ignore_index=True)
        y_dev = pd.concat([y_train, y_valid], ignore_index=True)
        print("Fitting the final ensemble on train + validation...")
        with mlflow.start_run(run_name="final-ensemble", nested=True) as final:
            mlflow.log_params({"model": "ExtraTrees V2 + MLP V0 (equal weight)",
                               "threshold": DECISION_THRESHOLD, "fit_on": "train + validation"})
            model = final_model(X_train, X_dev, y_dev)
            test_scores = evaluate(y_test, model.predict_proba(X_test)[:, 1])
            mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})
            model_uri = log_model(model, X_test).model_uri
            print(f"  test: {test_scores}")

        mlflow.set_tag("final_run_id", final.info.run_id)
        version, promoted = register_and_promote(
            model_uri, test_scores.get(SELECTION_METRIC), X_test, y_test, force_promote
        )
        mlflow.set_tags({"registered_version": version, "promoted": promoted})
    return parent.info.run_id


def register_joblib(path, if_empty=False):
    """Log and register an already-trained model file (no data needed)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if if_empty and current_production(MlflowClient()) is not None:
        print(f"{MODEL_NAME}@{MODEL_ALIAS} already exists; nothing to do.")
        return None

    model = joblib.load(path)
    example = pd.read_csv(ROOT / "tests" / "data" / "hotel_bookings_sample.csv").drop(
        columns=["is_canceled"]).head(3)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="register-committed-model"):
        mlflow.log_params({"source": str(path), "threshold": getattr(model, "threshold", None)})
        mlflow.set_tag("note", "Model trained by frontend/pipeline/train.py and committed to the repo; "
                               "metrics are in the README (not recomputed here).")
        model_uri = log_model(model, example).model_uri
    version, _ = register_and_promote(model_uri, None, None, None, force=True)
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="path to hotel_bookings.csv (full training mode)")
    parser.add_argument("--register-joblib", help="register an existing model file instead of training")
    parser.add_argument("--if-empty", action="store_true",
                        help="with --register-joblib: only register if no production model exists")
    parser.add_argument("--force-promote", action="store_true",
                        help="move the production alias even if the new model scores lower")
    args = parser.parse_args()

    if args.register_joblib:
        register_joblib(args.register_joblib, args.if_empty)
    elif args.data:
        train_and_register(args.data, args.force_promote)
    else:
        parser.error("give --data or --register-joblib")


if __name__ == "__main__":
    main()
