"""Shared runtime logging configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path


LOG_DIR_ENV = "LLM_CONTROL_PLANE_LOG_DIR"
LOG_LEVEL_ENV = "LLM_CONTROL_PLANE_LOG_LEVEL"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
TRACE_LOG_FILENAME = "traces.jsonl"
COMPONENT_LOGGERS = {
    "orchestrator": "src.orchestrator",
    "dashboard": "src.dashboard",
    "search": "src.search",
}


def get_log_dir() -> Path:
    """Return the configured runtime log directory."""
    raw_path = os.getenv(LOG_DIR_ENV, "").strip()
    return Path(raw_path).expanduser() if raw_path else DEFAULT_LOG_DIR


def get_component_log_path(component: str) -> Path:
    """Return the file path for a component log."""
    normalized = str(component or "").strip().lower()
    if normalized not in COMPONENT_LOGGERS:
        raise ValueError(f"Unknown log component: {component}")
    return get_log_dir() / f"{normalized}.log"


def get_trace_log_path() -> Path:
    """Return the structured trace JSONL path."""
    return get_log_dir() / TRACE_LOG_FILENAME


def _handler_exists(logger: logging.Logger, *, marker: str, path: Path) -> bool:
    resolved = str(path.resolve())
    for handler in logger.handlers:
        if getattr(handler, "_llm_control_plane_marker", None) != marker:
            continue
        if getattr(handler, "_llm_control_plane_path", None) == resolved:
            return True
    return False


def _add_file_handler(logger: logging.Logger, *, marker: str, path: Path) -> None:
    if _handler_exists(logger, marker=marker, path=path):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler._llm_control_plane_marker = marker
    handler._llm_control_plane_path = str(path.resolve())
    logger.addHandler(handler)


def _add_console_handler(root_logger: logging.Logger) -> None:
    marker = "console"
    if any(
        getattr(handler, "_llm_control_plane_marker", None) == marker
        for handler in root_logger.handlers
    ):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler._llm_control_plane_marker = marker
    root_logger.addHandler(handler)


def configure_logging() -> None:
    """Configure console and component file logging idempotently."""
    level_name = os.getenv(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    _add_console_handler(root_logger)

    for component, logger_name in COMPONENT_LOGGERS.items():
        component_logger = logging.getLogger(logger_name)
        component_logger.setLevel(level)
        _add_file_handler(
            component_logger,
            marker=f"{component}-file",
            path=get_component_log_path(component),
        )
