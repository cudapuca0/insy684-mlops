"""Train a quick model and save it in MLflow format to a local folder.

Used by the CI integration test so the API container can start without an
MLflow server:  python scripts/export_model.py artifacts/model
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mlflow.sklearn

from mlops_core.data import load_raw_data, split_data, validate_data
from mlops_core.modeling import build_pipeline, train


def main(out: str) -> None:
    train_df, _ = split_data(validate_data(load_raw_data()))
    model = train(build_pipeline("logistic_regression", {"C": 1.0}), train_df)
    out_path = Path(out)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlflow.sklearn.save_model(model, str(out_path))
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/model")
