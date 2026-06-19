# Dashboard Guide

The dashboard runs on `http://localhost:12341`, binds `127.0.0.1`, and communicates only with the proxy.

## Features

- Machine endpoint selector
- Dedicated model refresh action
- System prompt input
- RAG endpoint selector
- Search provider selector
- Dedicated RAG refresh action
- Streaming and non-streaming chat
- Reasoning effort selector
- File upload for inline prompt augmentation
- Single-Node workflow dispatch
- Workflow run creation, stepping, streaming, retry, clearing, and artifact inspection
- Conversation history viewer
- Runtime metadata panel

## Refresh Behavior

### Model Endpoint Selector

- Loads from proxy `/models`
- Has its own refresh action
- Refreshing it updates the endpoint list and hardware/model metadata view

### RAG Endpoint Selector

- Loads from `config.yaml`
- Health-checks each configured `health_url`
- Only healthy endpoints are displayed
- Has its own refresh action independent of model refresh

### Search Provider Selector

- Appears directly below the RAG selector
- Loads enabled providers from `config.yaml`
- Includes a persistent `None` option
- Refreshes whenever the RAG selector is refreshed
- Does not health-check providers; availability is config-driven

Workflow dispatch can change selector defaults:

- Single-Node `contextual_search`: enables the selector and defaults to the first configured non-`None` provider; `None` is removed when a real provider exists.
- Single-Node non-search workflows: resets to `None` and disables the selector.
- Workflows tab: selector remains enabled; `contextual_search` requires a real provider, while non-search workflows default to `None`.
- Graphs tab: manages LangGraph-native graph runs separately from workflows.

## Request Behavior

The dashboard sends concrete machine endpoint requests to `/{endpoint}`. Smart routing is a proxy API feature exposed at `/smart`; the dashboard does not maintain a separate local route pin.

When the user submits a message, the dashboard sends:

- selected machine endpoint
- optional `X-Convo-ID`
- optional `X-Reasoning-Effort`
- optional `X-RAG-Endpoint`
- optional system prompt as the first explicit system message
- optional turn-local search context injected as a synthetic `user` message when a search provider is selected

The dashboard does not inject retrieved context itself. It only selects the RAG endpoint; the proxy performs retrieval and request-body augmentation.

When a search provider is selected outside workflow dispatch, the dashboard first calls the proxy `POST /search/web` with `count: 5` and `use_reranker: false`, renders an inline search-candidate preface in the transcript, and adds one turn-local synthetic `user` message whose content is the ephemeral marker plus the proxy-provided `wrapped_results` JSON string. The proxy merges that ephemeral search message into the next real user turn before forwarding upstream and excludes it from durable history.

For Single-Node/ad hoc search, the proxy may use the configured query refiner to turn the latest request plus compact dashboard context into provider-ready query text or fanout queries. Workflow search can either use that query refiner for plain search prompts or let workflow LLM steps inspect the larger workflow context and dispatch planned queries directly through the selected provider. Workflow reranking is surfaced as its own workflow step.

When Single-Node workflow dispatch is enabled, the dashboard creates and runs a workflow for the submitted turn instead of sending a direct chat completion. The selected endpoint, reasoning effort, RAG endpoint, and search provider are copied into that workflow run.

System prompt and reasoning are first-turn conversation controls. If either changes after a conversation has started, the dashboard automatically forks to a new conversation id, copies prior durable user/assistant history behind the new system prompt, prints an explicit fork notice in the chat, and sends the next turn under the forked id.

## RAG Semantics

RAG context should usually not be merged into the dashboard’s editable system prompt field.

The current design is intentional:

- user system prompt: stable behavioral instructions
- proxy-rewritten latest user turn: turn-local grounded retrieval context

That avoids stale retrieval content persisting across turns and keeps prompt responsibilities separate.

## Runtime Panel

The runtime panel may show:

- elapsed time
- trace ID
- routing decision and workload type
- token usage and timings
- selected model and hardware metadata
- search provider, result count, degraded state, and warnings
- RAG endpoint, hit count, injection result, and skip reason

## Workflow Tab

The Workflows tab exposes the proxy workflow API. It can create a run from workflow params, selected endpoint, optional RAG endpoint, optional search provider, and optional uploaded UTF-8 context files. It can then advance one step, run to completion, stream run events, retry a step and its dependents, clear stored runs, and inspect step artifacts.

Search and rerank behavior follows workflow YAML, not the Single-Node ad hoc search path. For the built-in `contextual_search` workflow, the planner produces up to five provider queries and a separate context-resolved rerank query; the search step requests up to twenty results per query and the rerank step caps output at ten.

## Logout and Themes

- theme switching is handled through `shinyswatch`
- logout redirects to the configured Cloudflare Access logout URL

TL;DR: the dashboard is a thin operational UI over the proxy; the proxy owns search execution, workflow execution, retrieval, routing, history, and final request shaping.
