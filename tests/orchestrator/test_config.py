import subprocess
import sys
from pathlib import Path

import pytest

from src.orchestrator.config import MISSING_CONFIG_MESSAGE, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_missing_file_is_import_safe(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == {}


def test_load_config_required_missing_file_raises_clear_message(tmp_path):
    with pytest.raises(RuntimeError, match=MISSING_CONFIG_MESSAGE):
        load_config(tmp_path / "missing.yaml", required=True)


def test_llm_control_plane_missing_config_fails_cleanly(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "llm_control_plane.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 1
    assert MISSING_CONFIG_MESSAGE in result.stderr
    assert "FileNotFoundError" not in result.stderr
