import pytest

from mlops_core.config import load_config
from mlops_core.data import load_raw_data, split_data, validate_data
from mlops_core.modeling import build_pipeline, train


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    return validate_data(load_raw_data(cfg), cfg)


@pytest.fixture(scope="session")
def splits(dataset, cfg):
    return split_data(dataset, cfg)


@pytest.fixture(scope="session")
def trained_model(splits, cfg):
    train_df, _ = splits
    return train(build_pipeline("logistic_regression", {"C": 1.0}, cfg), train_df, cfg)
