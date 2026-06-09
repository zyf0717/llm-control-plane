from pathlib import Path

import pytest

from src import logging_config


def test_get_log_dir_defaults_to_repo_logs(monkeypatch):
    monkeypatch.delenv(logging_config.LOG_DIR_ENV, raising=False)

    assert logging_config.get_log_dir() == logging_config.REPO_ROOT / "logs"
    assert (
        logging_config.get_component_log_path("dashboard")
        == logging_config.REPO_ROOT / "logs" / "dashboard.log"
    )
    assert logging_config.get_trace_log_path() == (
        logging_config.REPO_ROOT / "logs" / "traces.jsonl"
    )


def test_get_log_dir_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(logging_config.LOG_DIR_ENV, str(tmp_path))

    assert logging_config.get_log_dir() == tmp_path
    assert logging_config.get_component_log_path("search") == tmp_path / "search.log"
    assert logging_config.get_trace_log_path() == tmp_path / "traces.jsonl"


def test_get_component_log_path_rejects_unknown_component():
    with pytest.raises(ValueError):
        logging_config.get_component_log_path("unknown")
