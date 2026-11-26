"""Load the project configuration (config.yaml) once and share it everywhere."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    # Containers mount config.yaml at /app/config.yaml and set CONFIG_PATH.
    return Path(os.environ.get("CONFIG_PATH", REPO_ROOT / "config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(_config_path(), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_root() -> Path:
    """Directory holding config.yaml; relative data paths resolve from here."""
    return _config_path().resolve().parent


def feature_columns(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])


def target_column(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["data"]["target"]
