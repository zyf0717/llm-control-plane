# Dashboard Guide

The dashboard runs on `http://localhost:12341` and communicates only with the proxy.

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

When a search provider is selected, the dashboard first calls the proxy `POST /search/web`, renders an inline search-candidate preface in the transcript, and adds one turn-local synthetic `user` message whose content is the ephemeral marker plus the proxy-provided `wrapped_results` JSON string. The proxy merges that ephemeral search message into the next real user turn before forwarding upstream and excludes it from durable history.

For Single-Node/ad hoc search, the proxy may use the configured query refiner to turn the latest request plus compact dashboard context into provider-ready query text or fanout queries. Workflow search does not use that query refiner; workflow LLM steps inspect the larger workflow context, decide the search queries, and dispatch them directly through the selected provider.

System prompt and reasoning are first-turn conversation controls. If either changes after a conversation has started, the dashboard automatically forks to a new `convo_id`, copies prior durable user/assistant history behind the new system prompt, prints an explicit fork notice in the chat, and sends the next turn under the forked id.

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

## Logout and Themes

- theme switching is handled through `shinyswatch`
- logout redirects to the configured Cloudflare Access logout URL

TL;DR: the dashboard is a thin operational UI over the proxy; the proxy owns search execution, retrieval, routing, history, and final request shaping.
