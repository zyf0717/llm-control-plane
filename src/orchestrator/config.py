"""Runtime configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_FILE = Path("config.yaml")
MISSING_CONFIG_MESSAGE = (
    "Missing config.yaml; copy config.example.yaml to config.yaml and edit endpoint URLs."
)


def load_config(
    config_path: str | Path = CONFIG_FILE,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Load YAML config, optionally enforcing presence for runtime launch."""
    path = Path(config_path)
    if not path.exists():
        if required:
            raise RuntimeError(MISSING_CONFIG_MESSAGE)
        return {}

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return config if isinstance(config, dict) else {}


def require_config_file(config_path: str | Path = CONFIG_FILE) -> None:
    """Fail clearly when the runtime config has not been created."""
    path = Path(config_path)
    if not path.exists():
        raise RuntimeError(MISSING_CONFIG_MESSAGE)
