# Dashboard Guide

The dashboard runs on `http://localhost:12341` and communicates only with the proxy.

## Features

- Model endpoint selector, including Auto mode
- Dedicated model refresh action
- System prompt input
- RAG endpoint selector
- Dedicated RAG refresh action
- Streaming and non-streaming chat
- Stop control for active streamed responses
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

## Request Behavior

When the user submits a message, the dashboard sends:

- selected model endpoint or Auto route
- optional `X-Convo-ID`
- optional `X-Reasoning-Effort`
- optional `X-RAG-Endpoint`
- optional system prompt as the first explicit system message

For streamed requests, the sidebar `Stop` control closes the active proxy stream. That preserves already-rendered partial output in the chat transcript and, when conversation history is enabled, the proxy persists the partial assistant text that was emitted before the stop.

The dashboard does not inject retrieved context itself. It only selects the RAG endpoint; the proxy performs retrieval and request-body augmentation.

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
- RAG endpoint, confidence, threshold, method, hit count, and injection result

## Logout and Themes

- theme switching is handled through `shinyswatch`
- logout redirects to the configured Cloudflare Access logout URL

TL;DR: the dashboard is a thin operational UI over the proxy; it selects model and RAG targets, but the proxy owns routing, retrieval, and final request shaping.
