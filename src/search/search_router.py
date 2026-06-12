"""Config-driven search provider routing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from .http_client import SearchHttpClient, SearchHttpClientConfig
from .normalize import dedupe_results
from .planner import SearchPlanner
from .search_cache import SearchCache
from .types import SearchArgs, SearchProvider, SearchResponse


@dataclass(slots=True)
class SearchProviderConfig:
    enabled: bool = False
    priority: int = 100
    fallback_only: bool = False
    cache_ttl_seconds: Optional[int] = None
    min_interval_seconds: float = 0.0


@dataclass(slots=True)
class SearchConfig:
    enabled: bool = True
    default_provider: str = "duckduckgo_html"
    timeout_ms: int = 7000
    max_results: int = 10
    max_total_results: int = 10
    cache_ttl_seconds: int = 900
    max_body_bytes: int = 2_500_000
    max_retries: int = 1

    def effective_count(self, requested_count: Optional[int]) -> int:
        if requested_count is None:
            return self.max_results
        return max(1, min(int(requested_count), self.max_results))

    def effective_ttl(self, args: SearchArgs, provider_config: SearchProviderConfig) -> int:
        if provider_config.cache_ttl_seconds is not None:
            return provider_config.cache_ttl_seconds
        if args.freshness == "day":
            return min(self.cache_ttl_seconds, 300)
        if args.freshness in {"week", "month"}:
            return min(self.cache_ttl_seconds, 1800)
        return self.cache_ttl_seconds


class SearchRouter:
    """Deterministic provider router with config-bound fallback."""

    def __init__(
        self,
        config: SearchConfig,
        *,
        providers: dict[str, SearchProvider],
        provider_configs: dict[str, SearchProviderConfig],
        http_client: Optional[SearchHttpClient] = None,
        cache: Optional[SearchCache] = None,
        planner: Optional[SearchPlanner] = None,
    ):
        self.config = config
        self.providers = providers
        self.provider_configs = provider_configs
        self.http_client = http_client or SearchHttpClient(
            SearchHttpClientConfig(
                timeout_ms=config.timeout_ms,
                max_body_bytes=config.max_body_bytes,
                max_retries=config.max_retries,
            )
        )
        self.cache = cache or SearchCache(default_ttl_seconds=config.cache_ttl_seconds)
        self.planner = planner
        self._last_request_at: dict[str, float] = {}
        self._rate_limit_lock = asyncio.Lock()

    async def search(self, args: SearchArgs) -> SearchResponse:
        self._validate_args(args)
        if not self.config.enabled:
            raise RuntimeError("Search module is disabled")

        warnings: list[str] = []
        original_query = args.query
        planner_metadata: dict[str, object] = {}
        query_args = [args]

        if self.planner is not None and args.use_planner:
            plan = await self.planner.plan(args)
            planner_metadata = plan.to_public_dict()
            if plan.warning:
                warnings.append(f"planner: {plan.warning}")

            updates: dict[str, object] = {}
            if plan.effective_query and plan.effective_query != args.query:
                updates["query"] = plan.effective_query
            if plan.freshness and plan.freshness != args.freshness:
                updates["freshness"] = plan.freshness
            if updates:
                args = replace(args, **updates)

            planned_queries = self._planned_queries(plan.queries, args.query)
            query_args = [replace(args, query=query) for query in planned_queries]

        explicit_provider = args.provider not in {None, "", "auto"}

        if len(query_args) == 1:
            response = await self._search_one_query(query_args[0], explicit_provider)
            if warnings:
                response.warnings = [*warnings, *response.warnings]
            self._attach_planner_metadata(
                response,
                original_query=original_query,
                effective_query=query_args[0].query,
                planner_metadata=planner_metadata,
            )
            return response

        branch_results = await asyncio.gather(
            *(self._search_one_query(query_arg, explicit_provider) for query_arg in query_args),
            return_exceptions=True,
        )
        responses: list[SearchResponse] = []
        for query_arg, branch_result in zip(query_args, branch_results):
            if isinstance(branch_result, Exception):
                if explicit_provider:
                    raise branch_result
                warnings.append(f"{query_arg.query}: {branch_result}")
                continue
            responses.append(branch_result)
            warnings.extend(
                f"{branch_result.query}: {warning}"
                for warning in branch_result.warnings
            )

        merged_results = dedupe_results(
            (result for response in responses for result in response.results),
            self.config.max_total_results,
        )
        primary_query = query_args[0].query

        if merged_results:
            provider_ids = {response.provider for response in responses if response.results}
            provider_id = next(iter(provider_ids)) if len(provider_ids) == 1 else "fanout"
            response = SearchResponse(
                query=primary_query,
                provider=provider_id,
                results=merged_results,
                degraded=bool(warnings),
                warnings=warnings,
            )
        else:
            response = SearchResponse(
                query=primary_query,
                provider="none",
                results=[],
                degraded=True,
                warnings=warnings,
            )

        self._attach_planner_metadata(
            response,
            original_query=original_query,
            effective_query=primary_query,
            planner_metadata=planner_metadata,
        )
        return response

    async def _search_one_query(
        self, args: SearchArgs, explicit_provider: bool
    ) -> SearchResponse:
        warnings: list[str] = []

        for provider in self._select_providers(args):
            provider_config = self.provider_configs[provider.id]
            await self._rate_limit(provider.id, provider_config)
            try:
                response = await provider.search(
                    args,
                    client=self.http_client,
                    cache=self.cache,
                    config=self.config,
                    provider_config=provider_config,
                )
            except Exception as exc:
                warnings.append(f"{provider.id}: {exc}")
                if explicit_provider:
                    raise
                continue

            if response.results:
                if warnings:
                    response.warnings = [*warnings, *response.warnings]
                return response

            warnings.append(f"{provider.id}: empty results")

        return SearchResponse(
            query=args.query,
            provider="none",
            results=[],
            degraded=True,
            warnings=warnings,
        )

    def _planned_queries(self, queries: list[str], fallback_query: str) -> list[str]:
        planned: list[str] = []
        seen: set[str] = set()
        for query in [*queries, fallback_query]:
            candidate = str(query or "").strip()
            key = " ".join(candidate.lower().split())
            if not candidate or key in seen:
                continue
            seen.add(key)
            planned.append(candidate)
        return planned or [fallback_query]

    def _attach_planner_metadata(
        self,
        response: SearchResponse,
        *,
        original_query: str,
        effective_query: str,
        planner_metadata: dict[str, object],
    ) -> None:
        if original_query != effective_query:
            response.original_query = original_query
        if planner_metadata:
            response.planner = planner_metadata

    def _select_providers(self, args: SearchArgs) -> list[SearchProvider]:
        explicit_provider = args.provider not in {None, "", "auto"}
        if explicit_provider:
            provider_id = str(args.provider)
            if provider_id not in self.providers:
                raise ValueError(f"Unknown search provider: {provider_id}")
            provider_config = self.provider_configs.get(provider_id)
            if provider_config is None or not provider_config.enabled:
                raise ValueError(f"Search provider is disabled: {provider_id}")
            return [self.providers[provider_id]]

        enabled: list[tuple[SearchProviderConfig, SearchProvider]] = []
        fallbacks: list[tuple[SearchProviderConfig, SearchProvider]] = []
        for provider_id, provider in self.providers.items():
            provider_config = self.provider_configs.get(provider_id)
            if provider_config is None or not provider_config.enabled:
                continue
            target = fallbacks if (provider_config.fallback_only or provider.fallback_only) else enabled
            target.append((provider_config, provider))

        enabled.sort(key=lambda item: item[0].priority)
        fallbacks.sort(key=lambda item: item[0].priority)
        ordered = [provider for _, provider in [*enabled, *fallbacks]]
        if self.config.default_provider in self.providers:
            ordered.sort(key=lambda provider: provider.id != self.config.default_provider)
        return ordered

    async def _rate_limit(
        self, provider_id: str, provider_config: SearchProviderConfig
    ) -> None:
        wait_seconds = max(0.0, provider_config.min_interval_seconds)
        if wait_seconds == 0:
            return

        async with self._rate_limit_lock:
            now = time.monotonic()
            last = self._last_request_at.get(provider_id)
            if last is not None:
                remaining = wait_seconds - (now - last)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_request_at[provider_id] = time.monotonic()

    def _validate_args(self, args: SearchArgs) -> None:
        if not isinstance(args.query, str) or not args.query.strip():
            raise ValueError("query is required")
        if args.count is not None and int(args.count) < 1:
            raise ValueError("count must be >= 1")
