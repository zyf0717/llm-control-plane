from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import proxy_services as services


router = APIRouter()


@router.get("/repo-context/repos")
async def list_repo_context_repositories():
    try:
        client = services.get_repo_context_client()
        return {"repositories": client.list_repositories()}
    except Exception as exc:
        services.logger.warning("Failed to list repo-context repositories: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
