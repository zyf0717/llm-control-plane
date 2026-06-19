from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_REPO_CONTEXT_PROJECT_DIR = Path("/home/yifei/repos/repo-context")
DEFAULT_REPO_CONTEXT_REPOS_ROOT = Path("/home/yifei/repos")
DEFAULT_REPO_CONTEXT_MAX_TURNS = 6
DEFAULT_REPO_CONTEXT_TIMEOUT_SECONDS = 180.0
DEFAULT_REPO_CONTEXT_MAX_CONCURRENT = 2


@dataclass(frozen=True, slots=True)
class RepoContextConfig:
    enabled: bool = True
    project_dir: Path = DEFAULT_REPO_CONTEXT_PROJECT_DIR
    repos_root: Path = DEFAULT_REPO_CONTEXT_REPOS_ROOT
    command: list[str] = field(default_factory=list)
    default_max_turns: int = DEFAULT_REPO_CONTEXT_MAX_TURNS
    timeout_seconds: float = DEFAULT_REPO_CONTEXT_TIMEOUT_SECONDS
    max_concurrent: int = DEFAULT_REPO_CONTEXT_MAX_CONCURRENT

    def effective_command(self) -> list[str]:
        if self.command:
            return list(self.command)
        return [
            "uv",
            "run",
            "--project",
            str(self.project_dir),
            "repo-context",
            "explore",
        ]


@dataclass(frozen=True, slots=True)
class RepoContextResult:
    text: str
    json: dict[str, Any]
    metadata: dict[str, Any]


class RepoContextClient:
    def __init__(self, config: RepoContextConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(max(1, int(config.max_concurrent)))

    def list_repositories(self) -> list[str]:
        if not self.config.enabled:
            return []
        repos_root = self._resolved_repos_root()
        repos: list[str] = []
        for child in repos_root.iterdir():
            if not child.is_dir():
                continue
            try:
                self.resolve_repo(child.name)
            except ValueError:
                continue
            repos.append(child.name)
        return sorted(repos)

    def resolve_repo(self, repo_name: str) -> Path:
        name = _validate_repo_name(repo_name)
        repos_root = self._resolved_repos_root()
        repo_root = (repos_root / name).resolve(strict=True)
        try:
            repo_root.relative_to(repos_root)
        except ValueError as exc:
            raise ValueError(f"repo target escapes configured root: {name}") from exc
        if not repo_root.is_dir():
            raise ValueError(f"repo target is not a directory: {name}")
        return repo_root

    async def explore_repository(
        self,
        *,
        query: str,
        repo_name: str,
        max_turns: int | None = None,
    ) -> RepoContextResult:
        if not self.config.enabled:
            raise ValueError("repo-context integration is disabled")

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("repo-context query is required")

        repo_root = self.resolve_repo(repo_name)
        effective_max_turns = int(max_turns or self.config.default_max_turns)
        if effective_max_turns <= 0:
            raise ValueError("repo-context max turns must be positive")

        command = [
            *self.config.effective_command(),
            "--query",
            normalized_query,
            "--repo",
            str(repo_root),
            "--max-turns",
            str(effective_max_turns),
            "--format",
            "json",
        ]
        raw = await self._run_command(command)
        payload = _parse_json_payload(raw)
        return build_repo_context_result(
            payload,
            repo_name=repo_root.name,
            repo_root=repo_root,
            query=normalized_query,
        )

    async def _run_command(self, command: list[str]) -> str:
        async with self._semaphore:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"repo-context command not found: {command[0]}"
                ) from exc

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=max(0.1, float(self.config.timeout_seconds)),
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                raise TimeoutError(
                    f"repo-context command timed out after "
                    f"{self.config.timeout_seconds:g}s"
                ) from exc

        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            detail = _truncate_error(stderr_text or "no stderr")
            raise RuntimeError(
                f"repo-context command failed with exit code "
                f"{process.returncode}: {detail}"
            )
        return stdout.decode("utf-8", errors="replace")

    def _resolved_repos_root(self) -> Path:
        try:
            root = self.config.repos_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(
                f"repo-context repos_root does not exist: {self.config.repos_root}"
            ) from exc
        if not root.is_dir():
            raise ValueError(
                f"repo-context repos_root is not a directory: {self.config.repos_root}"
            )
        return root


def load_repo_context_config(config: dict[str, Any] | None) -> RepoContextConfig:
    section = config.get("repo_context") if isinstance(config, dict) else None
    data = section if isinstance(section, dict) else {}
    project_dir = Path(
        str(data.get("project_dir") or DEFAULT_REPO_CONTEXT_PROJECT_DIR)
    ).expanduser()
    repos_root = Path(
        str(data.get("repos_root") or DEFAULT_REPO_CONTEXT_REPOS_ROOT)
    ).expanduser()
    command = _command_list(data.get("command"))
    return RepoContextConfig(
        enabled=_bool_value(data.get("enabled"), default=True),
        project_dir=project_dir,
        repos_root=repos_root,
        command=command,
        default_max_turns=max(
            1, _int_value(data.get("default_max_turns"), DEFAULT_REPO_CONTEXT_MAX_TURNS)
        ),
        timeout_seconds=max(
            0.1,
            _float_value(
                data.get("timeout_seconds"),
                DEFAULT_REPO_CONTEXT_TIMEOUT_SECONDS,
            ),
        ),
        max_concurrent=max(
            1,
            _int_value(
                data.get("max_concurrent"),
                DEFAULT_REPO_CONTEXT_MAX_CONCURRENT,
            ),
        ),
    )


def build_repo_context_result(
    payload: dict[str, Any],
    *,
    repo_name: str,
    repo_root: Path,
    query: str,
) -> RepoContextResult:
    answer = str(payload.get("answer") or "").strip()
    warnings = [str(item) for item in payload.get("warnings") or []]
    citations = [
        _citation_label(item)
        for item in payload.get("citations") or []
        if isinstance(item, dict)
    ]
    lines = [
        f"Repo: {repo_name} ({repo_root})",
        f"Query: {query}",
    ]
    if citations:
        lines.append("Citations:")
        lines.extend(f"- {label}" for label in citations if label)
    if answer:
        lines.extend(["Answer:", answer])
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    metadata = {
        "kind": "repo_context",
        "repo_name": repo_name,
        "repo_root": str(repo_root),
        "turns_used": payload.get("turns_used"),
        "truncated": bool(payload.get("truncated", False)),
    }
    return RepoContextResult(
        text="\n".join(lines).strip(),
        json=payload,
        metadata=metadata,
    )


def _validate_repo_name(repo_name: str) -> str:
    name = str(repo_name or "").strip()
    if not name:
        raise ValueError("repo target is required")
    if name in {".", ".."}:
        raise ValueError(f"invalid repo target: {name}")
    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)
    if any(separator and separator in name for separator in separators):
        raise ValueError(f"repo target must be a direct child name: {name}")
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] != name:
        raise ValueError(f"repo target must be a direct child name: {name}")
    return name


def _parse_json_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("repo-context command returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("repo-context command returned non-object JSON")
    return payload


def _citation_label(citation: dict[str, Any]) -> str:
    path = str(citation.get("path") or "").strip()
    start = citation.get("start_line")
    end = citation.get("end_line")
    if not path:
        return ""
    if start is None:
        return path
    if end is None or end == start:
        return f"{path}:{start}"
    return f"{path}:{start}-{end}"


def _command_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_value(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate_error(text: str, *, limit: int = 2000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "... [truncated]"
