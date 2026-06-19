import sys
from pathlib import Path

import pytest

from src.orchestrator.repo_context import RepoContextClient, RepoContextConfig


def test_repo_context_lists_immediate_child_directories(tmp_path):
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "file.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "alpha" / "nested").mkdir()
    client = RepoContextClient(RepoContextConfig(repos_root=tmp_path))

    assert client.list_repositories() == ["alpha", "beta"]


def test_repo_context_rejects_non_child_repo_names(tmp_path):
    client = RepoContextClient(RepoContextConfig(repos_root=tmp_path))

    with pytest.raises(ValueError, match="direct child"):
        client.resolve_repo("../outside")


@pytest.mark.asyncio
async def test_repo_context_subprocess_success(tmp_path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    script = _write_script(
        tmp_path,
        """
import json
print(json.dumps({
    "query": "Find validation",
    "repo_root": "repo-a",
    "answer": "src/api.py:1",
    "citations": [{"path": "src/api.py", "start_line": 1, "end_line": 1}],
    "turns_used": 1,
    "truncated": False,
    "warnings": [],
}))
""",
    )
    client = RepoContextClient(
        RepoContextConfig(
            repos_root=tmp_path,
            command=[sys.executable, str(script)],
        )
    )

    result = await client.explore_repository(
        query="Find validation",
        repo_name="repo-a",
    )

    assert result.json["answer"] == "src/api.py:1"
    assert "Citations:\n- src/api.py:1" in result.text
    assert result.metadata["kind"] == "repo_context"
    assert result.metadata["repo_name"] == "repo-a"
    assert result.metadata["turns_used"] == 1


@pytest.mark.asyncio
async def test_repo_context_subprocess_nonzero_exit(tmp_path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    script = _write_script(
        tmp_path,
        """
import sys
print("bad config", file=sys.stderr)
raise SystemExit(3)
""",
    )
    client = RepoContextClient(
        RepoContextConfig(
            repos_root=tmp_path,
            command=[sys.executable, str(script)],
        )
    )

    with pytest.raises(RuntimeError, match="exit code 3: bad config"):
        await client.explore_repository(query="Find validation", repo_name="repo-a")


@pytest.mark.asyncio
async def test_repo_context_subprocess_missing_command(tmp_path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    client = RepoContextClient(
        RepoContextConfig(
            repos_root=tmp_path,
            command=["missing-repo-context-binary-for-test"],
        )
    )

    with pytest.raises(RuntimeError, match="command not found"):
        await client.explore_repository(query="Find validation", repo_name="repo-a")


@pytest.mark.asyncio
async def test_repo_context_subprocess_timeout(tmp_path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    script = _write_script(
        tmp_path,
        """
import time
time.sleep(5)
""",
    )
    client = RepoContextClient(
        RepoContextConfig(
            repos_root=tmp_path,
            command=[sys.executable, str(script)],
            timeout_seconds=0.1,
        )
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await client.explore_repository(query="Find validation", repo_name="repo-a")


@pytest.mark.asyncio
async def test_repo_context_subprocess_invalid_json(tmp_path):
    repo = tmp_path / "repo-a"
    repo.mkdir()
    script = _write_script(
        tmp_path,
        """
print("not json")
""",
    )
    client = RepoContextClient(
        RepoContextConfig(
            repos_root=tmp_path,
            command=[sys.executable, str(script)],
        )
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        await client.explore_repository(query="Find validation", repo_name="repo-a")


def _write_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "repo_context_stub.py"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path
