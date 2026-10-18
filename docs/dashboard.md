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
- Single-Slot workflow dispatch
- Multi-Slot concurrent conversations on one endpoint, with optional burst submission
- Repo-context repository selection for workflows that declare `repo_name`
- Workflow run creation, stepping, streaming, retry, clearing, and artifact inspection
- Conversation history viewer
- System hardware panel and per-response diagnostics

## Refresh Behavior

### Model Endpoint Selector

- Loads from proxy `/models`
- Has its own refresh action
- Refreshing it updates the endpoint list and system hardware information

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

- Single-Slot workflow dispatch disables irrelevant right-panel controls based on the selected workflow spec.
- Workflows with `search` steps keep Search Provider enabled; `threaded_search` also defaults to the first configured non-`None` provider and removes `None` when a real provider exists.
- Workflows with `retrieval` steps keep Retrieval Endpoint enabled and default to the first configured non-`None` endpoint; `None` is removed when a real endpoint exists.
- Workflows with `repo_context` steps or `repo_name` params keep Repo Context Repository enabled.
- Workflows with `uploaded_source_text` params keep the upload controls enabled.
- Workflows tab: selectors remain enabled; `threaded_search` requires a real provider, while non-search workflows default search to `None`.
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

For Single-Slot/ad hoc search, the proxy may use the configured query refiner to turn the latest request plus compact dashboard context into provider-ready query text or fanout queries. Workflow search can either use that query refiner for plain search prompts or let workflow LLM steps inspect the larger workflow context and dispatch planned queries directly through the selected provider. Workflow reranking is surfaced as its own workflow step.

When Single-Slot workflow dispatch is enabled, the dashboard creates and runs a workflow for the submitted turn instead of sending a direct chat completion. The selected endpoint and reasoning effort are copied into that workflow run. Enabled right-panel controls are also copied into the run payload; disabled controls are ignored.

For the built-in `repo_context` workflow, selecting a non-`None` Repo Context Repository on Single-Slot automatically selects Workflow Dispatch `repo_context`; dispatch maps the submitted message to `query` and the selected repository to `repo_name`.

System prompt and reasoning are first-turn conversation controls. If either changes after a conversation has started, the dashboard automatically forks to a new conversation id, copies prior durable user/assistant history behind the new system prompt, prints an explicit fork notice in the chat, and sends the next turn under the forked id.

Single-Slot replies include the same response metrics footer as Multi-Slot. **Show
response metrics**, directly below **Show reasoning** in the sidebar, is on by default
and controls existing and future footers. Metrics are display-only and are excluded
from persisted conversation history. Non-streaming replies show `—` for time to first
token because token arrival cannot be observed. Workflow token counts and throughput
refer to the final response step; total elapsed time includes workflow setup and steps.

## Multi-Slot Tab

Multi-Slot provides 1–8 independent chat panes against one concrete endpoint, with four
panes shown initially. Send messages in individual panes to exercise asynchronous
usage; replies stream concurrently without blocking other panes. Each pane shows its
request status and elapsed time, including any upstream queue time.

Each response ends with a metrics footer showing time to first token (including
reasoning), total elapsed time, input/output tokens, then server-reported **Prefill**
and **Generation** throughput in tok/s. Unavailable values show `—`. Expand **Details**
for reported reasoning tokens, cached input tokens, MTP draft acceptance, model
information, and upstream slot ID. MTP acceptance uses accepted/proposed head tokens;
endpoints reporting only aggregate draft counts show **Draft acceptance** instead.
Zero proposed tokens show `—`, rather than a percentage. **Show response
metrics** under Generation settings is on by default and controls both existing and
future footers. Metrics are kept out of the conversation sent to the model.

Generation settings are shared: reasoning defaults to `medium`, while blank temperature
and maximum output token controls use upstream defaults. Temperature zero is sent
explicitly. Settings changes affect new requests, including every request in the next
burst; active requests keep their original settings.

Expand **Burst test** to send a shared prompt once to every visible idle pane. Each
request includes that pane's own history. Busy panes are skipped and individual drafts
are preserved. **Stop all** cancels dashboard streams; whether inference stops
immediately depends on the upstream server. Reset chats, endpoint refresh, endpoint
changes, and pane-count changes are disabled while requests are running.

Histories live only in the dashboard session. Hidden panes retain their history;
**Reset chats**, changing endpoints, or starting a new dashboard session clears it.
Failed and cancelled replies remain visible but are excluded from subsequent request
history. These requests omit conversation IDs and explicit slot IDs, so proxy
conversation persistence and affinity do not apply. Pane numbers identify conversations;
the upstream server assigns physical slots and may queue requests beyond its capacity.

## RAG Semantics

RAG context should usually not be merged into the dashboard’s editable system prompt field.

The current design is intentional:

- user system prompt: stable behavioral instructions
- proxy-rewritten latest user turn: turn-local grounded retrieval context

That avoids stale retrieval content persisting across turns and keeps prompt responsibilities separate.

## Endpoint Hardware and Response Details

**Endpoint Hardware** shows the selected endpoint's GPU, VRAM, SoC, CPU, and RAM when
available. It updates on endpoint selection and model refresh. Message submission,
streaming tokens, and response completion do not refresh this panel.

Turn-specific information belongs to each response's footer. The summary shows
elapsed time, TTFT, tokens, prefill, and generation rates. **Details** holds model
configuration, cache reuse, draft acceptance, trace ID, actual endpoint and workload,
workflow/run status, and search/RAG diagnostics when applicable. Model configuration
is captured from the matching endpoint's model catalog and response metadata for
that turn; changing endpoints or refreshing models does not rewrite earlier footers.
Raw usage, timing, runtime, and cache dumps are omitted to avoid repeating the same
values. Search candidate links remain in the transcript.

## Workflow Tab

The Workflows tab exposes the proxy workflow API. It can create a run from workflow params, selected endpoint, optional RAG endpoint, optional search provider, optional repo-context repository, and optional uploaded UTF-8 context files. It can then advance one step, run to completion, stream run events, retry a step and its dependents, clear stored runs, and inspect step artifacts.

The Repo Context Repository selector is populated from `GET /repo-context/repos`. If a selected workflow schema includes `repo_name`, the selector fills that param only when the editable Params JSON omits it or leaves it blank.

Search, retrieval, and rerank behavior follows workflow YAML, not the Single-Slot ad hoc paths. For the built-in `threaded_search` workflow, the planner produces up to five provider queries and a separate context-resolved rerank query; the search step requests candidates per query and the rerank step caps output according to workflow config. For `threaded_rag`, the planner produces one to six retrieval queries, dispatches them concurrently through the selected RAG endpoint, and synthesizes from the merged retrieval context.

## Connections and Diagnostics

The dashboard exchanges a heartbeat every 20 seconds to keep proxy connections active.
WebSocket close codes and server-side connection errors are logged in `logs/dashboard.log`.
Upstream stream exceptions are logged in `logs/orchestrator.log` and reported in the affected pane.

## Logout and Themes

- theme switching is handled through `shinyswatch`
- logout redirects to the configured Cloudflare Access logout URL

TL;DR: the dashboard is a thin operational UI over the proxy; the proxy owns search execution, workflow execution, retrieval, routing, history, and final request shaping.
