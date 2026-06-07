# Dashboard Guide

The dashboard runs on `http://localhost:12341` and communicates only with the proxy.

## Features

- Model endpoint selector, including Auto mode
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

When the user submits a message, the dashboard sends:

- selected model endpoint or Auto route
- optional `X-Convo-ID`
- optional `X-Reasoning-Effort`
- optional `X-RAG-Endpoint`
- optional system prompt as the first explicit system message
- optional turn-local search context injected as a synthetic `system` message when a search provider is selected

The dashboard does not inject retrieved context itself. It only selects the RAG endpoint; the proxy performs retrieval and request-body augmentation.

When a search provider is selected, the dashboard first calls the proxy `POST /search/web`, renders an inline search-candidate preface in the transcript, and injects only the proxy-provided `wrapped_results` payload into the outgoing request for that turn.

## RAG Semantics

RAG context should usually not be merged into the dashboard’s editable system prompt field.

The current design is intentional:

- user system prompt: stable behavioral instructions
- proxy-injected RAG message: turn-local retrieval context

That avoids stale retrieval content persisting across turns and keeps prompt responsibilities separate.

## Runtime Panel

The runtime panel may show:

- elapsed time
- routing decision and workload type
- token usage and timings
- selected model and hardware metadata
- search provider, result count, degraded state, and warnings
- RAG endpoint, confidence, threshold, method, hit count, and injection result

## Logout and Themes

- theme switching is handled through `shinyswatch`
- logout redirects to the configured Cloudflare Access logout URL

TL;DR: the dashboard is a thin operational UI over the proxy; it selects model, RAG, and optional search targets, while the proxy still owns search execution, retrieval, routing, and final request shaping.
