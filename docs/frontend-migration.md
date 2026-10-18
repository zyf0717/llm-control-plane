# Frontend migration architecture

Status: proposed architecture and implementation sequence, 18 October 2026.

Replace the Shiny dashboard with a TypeScript browser application. Retain FastAPI,
the Python orchestration and search code, and the existing SQLite stores. The
recommended frontend is **React with Vite**, with browser routing and a small set
of established libraries. Production serves compiled assets alongside the Python
API; Node is a development and build dependency.

Evaluate existing LLM frontends before building chat components. The preferred
reuse candidate is assistant-ui inside the proposed operator console; LibreChat
and big-AGI are the leading complete-application alternatives. The assessment
below separates component reuse, companion applications, and broader platform
adoption because they have different backend and deployment costs.

The first migration task is to extract application behavior currently embedded in
Shiny. Changing the rendering framework alone would move conversation policy,
prompt construction, and transcript persistence into browser code and duplicate
backend responsibilities.

This proposal assumes an operator-facing application, a small trusted user group,
one backend process, and no requirement for public search indexing or server-side
rendering. Revisit deployment and authorization if independent user accounts or
multiple backend workers become requirements. Implementation has not started.

## Current architecture and migration drivers

The current entry point starts FastAPI on `0.0.0.0:12340` and Shiny on
`127.0.0.1:12341` in daemon threads. There is no checked-in frontend build or
deployment configuration. The dashboard contains about 6,400 lines of Python;
`app_server.py` alone contains 2,312 lines and `app_ui.py` contains 679.

The existing HTTP boundary is useful, but incomplete:

| Existing behavior | Current location | Consequence for migration |
|---|---|---|
| Routing, reasoning normalization, retrieval injection, persistent history, slot affinity | `src/orchestrator/request_processor.py`, `upstream_proxy.py`, `conversation_store.py` | Preserve as backend authority. |
| Search providers, model-assisted refinement and reranking | `src/search/`, `src/orchestrator/search_routes.py` | Preserve implementation and search API. |
| Workflow and graph registries, execution, storage, run APIs | `src/orchestrator/workflow/`, `graph/`, `runtime/`, `src/graphs/` | Preserve separate domain models and APIs. |
| Configuration reads, retrieval health probes, provider choices and display timezone | `src/dashboard/utils.py` | Expose a filtered backend catalog; browsers cannot read local YAML or probe server-local services. |
| Trace viewer reads local JSONL files | `src/dashboard/utils.py:read_trace_events` | Add a bounded backend query endpoint. |
| Cloudflare service credentials attached to dashboard HTTP calls | `src/dashboard/chat_client.py` and other clients | Keep service credentials in Python; define browser authentication independently. |
| Prompt/reasoning changes fork conversations and copy prior history | `src/dashboard/app_server.py`, `prompt_state.py` | Move the operation into a backend transaction. |
| Ad hoc search evidence and workflow dispatch prepare chat requests | `src/dashboard/app_server.py`, `search_flow.py`, `workflow_server_helpers.py` | Extract Python application functions before replacing UI handlers. |
| Workflow chat dispatch appends user and final assistant messages | `src/dashboard/app_server.py` | Backend must finalize the transcript even when the browser leaves. |
| Response diagnostics combine raw metadata, model catalog snapshots and formatting | `response_metrics.py`, `cache_telemetry.py` | Keep normalization in Python and rewrite rendering in TypeScript. |
| Up to eight independent chats, burst submission and stop controls | `src/dashboard/multi_slot.py` | Reimplement browser state and concurrency while retaining session-only histories. |

Documentation and code currently disagree on some contracts. The code uses
`X-Conversation-ID`, `X-Retrieval-Endpoint`, and `X-Retrieval-*`; existing guides
also describe `X-Convo-ID` and `X-RAG-*`. Retrieval discovery currently logs failed
health probes but still includes those endpoints, despite the dashboard guide
describing healthy-only choices. Establish fixtures from running code and tests,
then correct these descriptions during contract work.

## Existing LLM frontends and reuse options

Assessment date: 18 October 2026. Capabilities, license files and repository status
were checked against upstream sources; no candidate has been integrated or
benchmarked against this backend. Fit and effort judgments below apply to this
repository. Record the selected release/commit and its license when prototyping.

### Chat and agent interfaces

| Project and license | Existing frontend capability | Integration cost and suitable role |
|---|---|---|
| [assistant-ui](https://github.com/assistant-ui/assistant-ui), [MIT](https://github.com/assistant-ui/assistant-ui/blob/main/LICENSE) | React chat components and runtime adapters. Its [ExternalStoreRuntime](https://www.assistant-ui.com/docs/runtimes/custom/external-store) accepts application-owned messages and callbacks. This is a component library, not a complete operator application. | **First reuse candidate.** Connect the planned stream reducer and Python APIs through an adapter. Keep backend history and fork policy authoritative; disable edit/regenerate actions until corresponding backend operations exist. Build catalog, multi-slot coordination, workflow, graph and trace views locally. No additional chat database or hosted service is required by this integration approach. |
| [LibreChat](https://github.com/LibreChat-AI/LibreChat), [MIT](https://github.com/LibreChat-AI/LibreChat/blob/main/LICENSE) | Complete React chat application with model selection, reasoning display, files, branching, agents and artifacts. [Custom endpoints](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/custom_endpoint) support configurable base URLs, direct completion URLs and headers. | **Leading general chat application.** Its standard deployment includes a Node API and MongoDB, with additional services in the [default Compose stack](https://github.com/LibreChat-AI/LibreChat/blob/main/docker-compose.yml). Use as a companion through a compatibility adapter, or accept a substantial frontend fork to share this project's history and run views. Adopting its agents does not expose this project's YAML workflow or graph lifecycle automatically. |
| [big-AGI Open](https://github.com/enricoros/big-AGI), [MIT](https://github.com/enricoros/big-AGI/blob/main/LICENSE) | React/Next.js multi-model workspace with Beam comparison/merge, personas, streaming and request inspection. The open repository is distinct from the hosted service's complete feature set. | **Leading experiment/comparison application.** Its concurrent model UX is relevant to Multi-Slot, but Beam comparison/synthesis does not establish parity with eight independent same-endpoint pane histories and burst tests. Evaluate as a companion or a source of reusable interaction patterns; adapting provider transport, history, metrics and run management remains work. Retaining the application also retains its Next.js deployment model. |
| [Hugging Face Chat UI](https://github.com/huggingface/chat-ui), [Apache-2.0](https://github.com/huggingface/chat-ui/blob/main/LICENSE) | SvelteKit application powering HuggingChat. Current upstream supports an OpenAI-compatible base URL and model discovery; chat history, users, files and settings use MongoDB, including an embedded development option. | **Leading complete Svelte alternative.** Useful if a chat-first product and Svelte maintenance are preferred. It brings server routes and its own persistence; serving only its static assets from FastAPI is not a configuration change. Provider/retrieval selection, this project's diagnostics and both run consoles need adaptation. |
| [LangChain Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui), [MIT](https://github.com/langchain-ai/agent-chat-ui/blob/main/LICENSE) | Next.js chat application for a LangGraph server exposing a `messages` state key, using the LangGraph SDK and assistant/thread/run concepts. | **Graph-chat specialist.** This project's `/graphs` and `/graph-runs` APIs implement a different contract; several built-in graphs accept task-specific inputs such as `question`, not a chat `messages` state. Reuse needs a protocol/state adapter or a separately operated LangGraph server. It does not replace YAML workflow management, endpoint operations or multi-slot experiments. |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm), [MIT](https://github.com/Mintplex-Labs/anything-llm/blob/master/LICENSE) | React/Vite workspace chat, document ingestion, citations, agents and provider configuration. The repository also includes a Node/Express server and document collector. | **Document/RAG application alternative.** Suitable if workspace/document management becomes the product focus. Keeping this project's retrieval and orchestration requires explicit ownership boundaries; adopting AnythingLLM's ingestion, routing or agents would also migrate backend responsibilities. Extracting its frontend alone is substantial API adaptation. |

The preferred integration for assistant-ui is a small adapter over our feature
state, with one runtime per active pane. Durable chat keeps stable backend
conversation IDs; ephemeral multi-slot panes continue to omit those IDs.
Keep structured reasoning, evidence and immutable metrics as message parts or
custom components. Test whether its message grouping, branching model and runtime
lifecycle preserve our transcript boundaries. Its optional cloud persistence and
standard provider runtimes are not needed for this route. Keep the existing
backend-extraction work regardless of which chat renderer wins.

### Gateway consoles and workflow builders

These projects address other parts of a control plane. Their consoles are coupled
to their own management APIs, so adopting the full product changes the backend
scope. They are useful comparisons when deciding how much of this project should
remain custom.

| Project | Existing operational UI | Architectural implication |
|---|---|---|
| [LiteLLM Admin UI](https://docs.litellm.ai/docs/proxy/ui) | Model configuration, virtual keys, usage/spend and a [gateway-backed chat page](https://docs.litellm.ai/docs/proxy/ui/chat_ui). Its [license](https://github.com/BerriAI/litellm/blob/main/LICENSE) is MIT outside the separately licensed enterprise directory. | Closest alternative if gateway administration becomes the primary requirement. The UI expects LiteLLM's gateway, database and management APIs; making it operate this Python proxy requires those adapters. Adopting the gateway itself is a separate backend migration, with this project's workflow/graph services still requiring a home. |
| [Bifrost](https://github.com/maximhq/bifrost), [Apache-2.0](https://github.com/maximhq/bifrost/blob/dev/LICENSE) | Built-in gateway/provider configuration UI alongside routing, failover and caching. | A second relevant gateway-console comparison. The integrated UI expects Bifrost's gateway/configuration APIs. Evaluate whole-gateway adoption separately from copying views; it does not establish compatibility with our conversation pins, retrieval policy, run stores or per-call metrics. |
| [Langflow](https://github.com/langflow-ai/langflow), [MIT](https://github.com/langflow-ai/langflow/blob/main/LICENSE) | Visual agent/workflow authoring, a testing playground and API/MCP execution. | Appropriate if visual workflow authoring becomes a requirement. Its flow definitions and runtime are a different model from our YAML DAGs and LangGraph subsystem. Adopting it is a workflow-platform decision; using only its editor would still require a deliberate schema and execution adapter. |

### License and maintenance constraints

Keep these candidates visible, but distinguish their additional terms and
maintenance status from the permissively licensed shortlist:

- **[Open WebUI](https://github.com/open-webui/open-webui):** a relevant complete
  chat/RAG application with [OpenAI-compatible connections](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)
  and [Functions/Pipelines](https://docs.openwebui.com/features/extensibility/).
  It could serve as a companion through the adapter described below. Its current
  [Open WebUI License](https://github.com/open-webui/open-webui/blob/main/LICENSE)
  includes a branding-preservation condition with exceptions. Treat it as a
  custom-license option and assess the exact terms for a branded fork; do not
  assume an unrestricted MIT/BSD license from older descriptions. Its own backend,
  history and RAG integrations also remain part of full-application adoption.
- **[LobeHub, formerly LobeChat](https://github.com/lobehub/lobehub):** a broader
  agent/chat workspace. The [LobeHub Community License](https://github.com/lobehub/lobehub/blob/canary/LICENSE)
  adds conditions to Apache-2.0, including a commercial-license requirement for
  developing and distributing a derivative work. Assess it as a custom-license
  product option, with its own backend and agent model, rather than as a
  permissively licensed frontend starter.
- **[Chatbot UI](https://github.com/mckaywrigley/chatbot-ui):** MIT-licensed
  Next.js/Supabase chat application, but upstream repository metadata reports its
  last push in August 2024 at this assessment date. Its persistence stack and
  maintenance outlook make it a lower-priority migration base.
- **[Chainlit](https://github.com/Chainlit/chainlit)** and
  **[Open Agent Platform](https://github.com/langchain-ai/open-agent-platform)**
  are relevant conversational/agent UI references, but their upstream repositories
  are archived at this assessment date. Exclude them from the initial adoption
  shortlist unless a maintained successor is identified and evaluated.

### Compatibility work required by complete applications

OpenAI-compatible request bodies do not make the current proxy a complete
drop-in provider. `proxy.py` exposes `/models`, `/smart` and `/{endpoint}`;
`request_processor.py` resolves routing from the path and reads custom control
headers. There is no general `/v1/chat/completions` route that maps its `model`
field to a configured endpoint. A frontend expecting that route needs an adapter.

For a companion application, plan an additive Python compatibility facade with
`GET /v1/models` and `POST /v1/chat/completions`. Give each endpoint/model pairing
a stable public model alias, resolve that alias to the configured endpoint, and
forward the original upstream model ID through shared proxy functions. Preserve
streaming reasoning, usage and errors. Reserve `/v1` and check configured endpoint
names for collisions when enabling this optional facade. LibreChat's
direct-endpoint configuration can avoid some path adaptation for a fixed endpoint,
but does not solve shared history or management-view integration.

Choose conversation ownership explicitly:

1. **Companion owns chat history.** Let it send its normal full message list and
   omit `X-Conversation-ID`. This uses the proxy's stateless mode. Proxy-managed
   durable history, route pins and conversation slot affinity are then unavailable
   for those chats. Disable overlapping frontend-side retrieval/search/routing
   features when the control plane should remain responsible for those actions.
2. **Control plane owns chat history.** Implement an integration that maps stable
   conversation IDs, sends new turns only, and routes edits/forks through backend
   operations. Passing full-history messages together with `X-Conversation-ID`
   will trigger this proxy's replay rejection. This requires deeper frontend
   adaptation and reconciliation of any second conversation database.

Complete apps still need custom views/adapters for workflow advance/retry,
artifacts, graph resume, trace filtering, frozen response metrics and same-endpoint
burst testing. Their native tool/agent displays are not evidence of parity with
these management actions. Cost the compatibility layer, extra services, data
ownership, ongoing upstream merges and feature gaps alongside saved UI work.
No full application has been demonstrated to replace the entire dashboard here.

### Reuse decision before implementation

Add a **2–3 engineer-day reuse evaluation** to phase 0. This is planned work, not
an instruction to install or deploy these applications now.

1. Prototype assistant-ui against a deterministic backend stream with a normal
   two-turn conversation, separate reasoning/evidence/metrics, an explicit fork,
   and eight independent panes. Check that runtime adapters preserve server
   ownership and do not require a hosted service or an incompatible wire protocol.
2. Exercise a thin compatibility facade with LibreChat as the general-chat
   reference and big-AGI as the comparison reference. Use fixtures to inspect
   actual payloads, history replay, model aliases, custom metadata, errors and
   disconnect behavior. Check the self-hosted repository features rather than
   assuming hosted demos are fully available in the open edition.
3. Record a feature-by-feature result for chat, pane/burst tests, workflow actions,
   graph actions, metrics, traces and uploads. Compare required patches, runtime
   services, storage changes and release maintenance, then choose a reuse route.

Prefer **React/Vite plus assistant-ui components** if the adapter passes these
checks. Keep custom React chat components as the fallback. Prefer **LibreChat as a
companion** if a ready general chat product is sufficient; prefer **big-AGI as a
companion** when comparative model experiments dominate. A full application fork,
gateway replacement or workflow-platform adoption needs a revised architecture
and estimate. The remaining sections describe the component-reuse/custom-console
route and do not assume another application's backend can be removed unchanged.

## assistant-ui stack assessment

**Recommended role: the chat component and runtime layer inside the React
operator console.** It can replace ShinyChat and a substantial amount of custom
chat interaction code. The application shell, operational views, transport adapter
and Python services remain project-owned. This is a source/documentation
assessment; integration and eight-pane behavior remain prototype acceptance gates.

assistant-ui offers both ready-made styled elements and unstyled accessible
[primitives](https://www.assistant-ui.com/docs/primitives). Use the primitives
with the proposed CSS/theme for tighter control, or adopt selected styled
elements together with their actual styling dependencies. Avoid adding two
independent chat/composer implementations.

### Replacement map

| Existing or proposed stack area | What assistant-ui replaces | What this project still implements |
|---|---|---|
| `shinychat.chat_ui`, `Chat`, chat sections of `app_ui.py` and `app_server.py` | Transcript, message/composer widgets, message actions and their runtime bindings, using [Thread](https://www.assistant-ui.com/docs/primitives/thread) and [Composer](https://www.assistant-ui.com/docs/primitives/composer). | Endpoint/settings controls, submission policy, backend calls and the surrounding operator layout. Removing the rest of Shiny still requires migrating the other views. |
| Incremental message rendering and scroll/focus behavior | Runtime-driven message updates, viewport auto-scroll, composer interaction and accessible chat structure. | Decoding network frames, accumulating canonical request state, associating events with request IDs, and preserving interrupted/error outcomes. A runtime can display streams without providing this backend's transport. |
| Markdown, reasoning and search-result presentation | Standard text/Markdown rendering and message-part composition; custom reasoning/source renderers can use its message context. | Evidence cards, raw JSON view and exact visibility rules. [Bare message primitives](https://www.assistant-ui.com/docs/primitives/message) render reasoning and source parts only when renderers are supplied. Hidden workflow steps remain hidden. |
| `multi_slot.py` chat widget in each pane | One reusable thread/composer with a separate runtime scope per pane. | Pane grid/count, burst scheduling, busy guards, settings snapshots, preserved drafts, Stop all, and successful-history selection. A library thread list is not a concurrent eight-pane scheduler. |
| Upload controls and attachment chips | Attachment selection/display lifecycle through an [attachment adapter](https://www.assistant-ui.com/docs/guides/attachments). | UTF-8 reading and limits, filename/text payloads, removal cleanup, inline augmentation and workflow `uploaded_source_text` mapping. Built-in image/file support does not establish compatibility with the current text-upload contract. |
| History conversation selector and read-only transcript | [Thread-list primitives](https://www.assistant-ui.com/docs/primitives/thread-list) and message rendering can supply the UI. | Fetching existing conversation APIs, stable ID mapping, reload/reconciliation and backend fork operations. Expose archive/delete/rename only after corresponding backend support exists. |
| `response_metrics.py` footer rendering | Message metadata/context supplies a place to render a custom metrics component. | Existing Python normalization, frozen model configuration, per-call attribution, missing-value rules, cache diagnostics and a custom metrics footer. Its [timing hook](https://www.assistant-ui.com/docs/guides/message-timing) is experimental and does not replace this project's measurement definitions. |
| Workflow-backed chat in `app_server.py` | Common transcript/message-part UI for visible intermediate/final output. | Dispatch, run creation, step visibility, transcript publication, retries and execution ownership. Use custom data parts for workflow progress; displaying a tool card does not implement a workflow executor. |
| Workflow, graph and trace tabs; endpoint/retrieval/provider selectors | Shared chat output widgets can be reused within these views. | Catalog forms, timelines, artifacts, graph resume controls, trace filters and all domain-specific clients/actions. |
| React, Vite, browser routing, query cache and theme | Chat-specific runtime context removes some local UI wiring. | React/Vite, React Router, TanStack Query for server snapshots, CSS/theme, URL state and application-level coordination remain. |
| FastAPI, search, routing/retrieval, workflow/graph engines, SQLite and credentials | No backend replacement is proposed. | All existing domain services, persistence and upstream authentication remain in Python. |

### Runtime and transport choice

Use **ExternalStoreRuntime first** because the proposed frontend deliberately
owns request state, immutable diagnostics and multi-pane coordination. Its
[external-store API](https://www.assistant-ui.com/docs/api-reference/external-store/runtime)
adapts that state to chat widgets; it does not eliminate our reducer or transport.
The adapter supplies messages, running/error state and submit/cancel callbacks.
Keep composer UI state in the library and avoid a second copy of its input state
except where burst controls must preserve pane drafts.

**LocalRuntime is a valid simpler alternative**, including for a Python backend.
Its [ChatModelAdapter](https://www.assistant-ui.com/docs/runtimes/custom/local-runtime)
can call our HTTP API and yield incremental results while the runtime owns local
message state. It could replace more custom message/run bookkeeping, particularly
in ephemeral panes. For durable chat, the adapter must still send new turns only,
load canonical backend history, and route forks through Python. Compare it in the
same spike; choose it if those policies stay simple. Start with one runtime
strategy for both chat modes and split only if there is a clear reduction in code.

The runtime adapter can call `/api/ui/v1/chat/stream` directly. Retain `fetch`,
abort handling, the SSE parser and per-protocol event reducers in the browser API
client. assistant-ui's supported protocol/framework runtimes provide different
tradeoffs, but adopting AI SDK/DataStream/AssistantTransport would require a
deliberate protocol adapter; the current stream does not become compatible by
installing a package. The OpenAI `/v1` facade proposed for complete companion
applications is not required for this integration. The LangGraph runtime also
does not directly consume our custom `/graph-runs` contract.

```mermaid
flowchart LR
    UI[assistant-ui thread, composer and message parts] <--> RT[Runtime adapter]
    RT <--> ST[Project request and pane state]
    ST <--> HTTP[Typed fetch and SSE client]
    HTTP <--> API[FastAPI UI and existing domain APIs]
    API --> PY[Python routing, search, execution and SQLite]
```

For ExternalStoreRuntime, the minimal package scope is `@assistant-ui/react` and,
when using its Markdown renderer, `@assistant-ui/react-markdown`. The latter
[depends on react-markdown](https://github.com/assistant-ui/assistant-ui/blob/main/packages/react-markdown/package.json),
so it replaces our Markdown integration code rather than removing that renderer
from the dependency tree. Core package peer dependencies are
[React and React DOM](https://github.com/assistant-ui/assistant-ui/blob/main/packages/react/package.json);
the proposed integration can retain a static Vite frontend and Python-only
production server. No Vercel AI SDK or production Next.js server is required by
this runtime choice. Keep Assistant Cloud unconfigured and retain SQLite history.

### Adoption conditions

Add these checks to the existing reuse spike, without increasing its scope to a
full frontend migration:

- Preserve separate search evidence, workflow outputs and final answers.
  External message conversion can join adjacent assistant messages; select an
  explicit join policy and verify that message IDs, metadata and metrics remain
  associated with the right output. Diagnostic data must not enter model inputs.
- Keep edit/regenerate/branch controls disabled until they invoke real backend
  operations. The library's alternate-answer branches and our system-prompt/
  reasoning conversation forks represent different state changes.
- Keep runtime instances and active request state alive across route changes and
  hidden panes. Eight panes must maintain separate input, messages, running state,
  request IDs and abort controllers. Ignore late events from aborted requests.
- Map cancellation to **Stop receiving** under current backend semantics.
  Cancelling a runtime or fetch does not establish that inference or a run stopped.
- Verify explicit reasoning/source rendering, upload augmentation, raw JSON,
  preserved zero/missing metrics and backend history reload. Observe browser
  network requests to verify the selected configuration uses only intended APIs.
- Pin the tested package versions. Prefer documented stable entry points over
  experimental timing, persistence or internal runtime APIs. Treat registry
  component code copied into the project as code we must maintain.

Adopt if these checks pass with a small adapter and custom message renderers.
The expected saving is chat widget, interaction and rendering work; backend
extraction, execution lifecycle fixes and operational views remain on the critical
path. Keep the existing estimate until a working slice establishes actual savings.

## Framework decision

Framework suitability here depends on managing concurrent streams, rich diagnostic
views, explicit state, browser tests, and a Python API. All three mainstream
browser component frameworks can meet those requirements; there is no measured
performance advantage established by this assessment.

| Option | Fit and tradeoff | Decision |
|---|---|---|
| React + TypeScript + Vite | Explicit reducers and broad component integration choices; requires deliberate routing, query-cache and effect conventions. Static build fits the existing Python backend. | Recommended default. |
| Svelte + TypeScript, with SvelteKit static SPA output | Concise component/state code and integrated routing. Equally viable if maintainers prefer Svelte; the same backend extraction is still necessary. | Strong alternative. |
| Vue + TypeScript + Vite | Suitable SPA component model and routing ecosystem; choose it if maintainers have stronger Vue experience. | Viable alternative. |
| Next.js | Supports static export, so it does not inherently require a production Node server. Its server rendering and server-component features do not address the current requirements. | Defer unless server-rendered product pages or a deliberate JS server become requirements. |
| Further Shiny refactoring or another Python dashboard framework | Can improve module size, but retains server-driven browser interaction and the need to integrate custom browser behaviors. A lateral framework switch would also rewrite UI code. | Keep Shiny for transition; replace it at cutover. |

These are engineering judgments for this repository. React documents both the
[Vite SPA approach and its routing/data-fetching tradeoffs](https://react.dev/learn/build-a-react-app-from-scratch).
The alternatives support [SvelteKit SPAs](https://svelte.dev/docs/kit/single-page-apps),
[Vue SPAs](https://vuejs.org/guide/extras/ways-of-using-vue.html), and
[Next.js static export](https://nextjs.org/docs/app/guides/static-exports).

Use this initial frontend stack for the component-reuse/custom-console route:

| Concern | Choice |
|---|---|
| Components and types | React, TypeScript with strict checking. |
| Build and local development | Vite; one `frontend/package.json` and npm lockfile. Pin a supported Node LTS when scaffolding. |
| Navigation | React Router in [declarative mode](https://reactrouter.com/start/modes), based at `/ui`; route-level lazy loading. |
| Server snapshots and catalogs | TanStack Query. Disable focus/reconnect refetch for model discovery and disable automatic retries for commands. Streams have their own lifecycle. |
| Local state | Feature reducers and small context providers; add a global state library only if measured complexity warrants it. |
| Components and styling | Semantic HTML, CSS variables and CSS modules; accessible library primitives only where needed for dialogs, menus or tabs. |
| Model output | assistant-ui chat primitives and `@assistant-ui/react-markdown` if the reuse evaluation passes; otherwise a direct `react-markdown` integration. Keep raw HTML disabled and reasoning, evidence, errors and metrics separate. |
| API types | [openapi-typescript](https://openapi-ts.dev/introduction) for documented HTTP contracts; explicit schemas/fixtures for stream frames and dynamic workflow/graph payloads. |
| Streaming | `fetch`, `AbortController`, and an established SSE parser such as [eventsource-parser](https://github.com/rexxars/eventsource-parser). |
| Verification | Vitest and Testing Library for behavior; Playwright for browser flows against deterministic backend fixtures. |

Model discovery has side effects on the router's reachable-endpoint cache. Treat
refresh as deliberate rather than a consequence of tab focus. TanStack Query
[supports disabling focus refetch](https://tanstack.com/query/latest/docs/framework/react/guides/window-focus-refetching).
Do not adopt a chat SDK that replaces the existing completion/event protocol
unless it preserves the tested reasoning, usage, routing and error semantics.

## Target boundaries and deployment

```mermaid
flowchart TD
    B[Browser: React application] -->|same-origin HTTP and SSE| F[FastAPI application]
    C[Existing API clients] -->|existing routes and headers| F
    F --> U[Typed UI routes: catalog, turns, fork, traces]
    F --> A[Existing chat, search, workflow and graph routes]
    U --> P[Python application functions]
    A --> P
    P --> R[Request processing and upstream proxy]
    P --> W[Workflow executor and store]
    P --> G[Graph executor and store]
    P --> S[Search service]
    P --> D[(Existing SQLite stores and trace log)]
    R --> L[Configured LLM and retrieval endpoints]
    S --> E[Configured search providers]
    F --> V[Static frontend assets at /ui]
```

Keep one Python backend and one frontend project in this repository. UI routes are
thin adapters over Python functions in the existing process. They do not call the
proxy over HTTP or import Shiny. Workflow and graph execution stay independent;
shared transport and metric components do not imply a shared executor.

In production, serve `frontend/dist` under `/ui/`. Register UI APIs, existing APIs,
and the static mount before the current `/{path:path}` proxy fallback. Restrict
SPA fallback to known UI navigation requests: missing assets return 404 and API
errors never return the frontend HTML. Reserve `/ui` and `/api/ui/v1` explicitly
and check configured endpoint names for conflicts. Vite supports a
[static production build with a configured base path](https://vite.dev/guide/build).

During development, run FastAPI, legacy Shiny, and Vite separately. Vite proxies
the explicitly used API paths to FastAPI; its browser URLs remain same-origin.
Do not proxy the entire root path, which belongs to Vite's application server.
At cutover, the normal Python entry point starts the backend with compiled assets.
Keep an explicit legacy launch option until rollback is retired. Production starts
from an existing asset build and never installs npm packages at startup.

The existing bind addresses matter: placing the UI on port 12340 would expand its
reach beyond the current localhost-only dashboard. UI-enabled local deployments
must bind to loopback by default; remote operator access must put the complete UI
and its APIs behind the existing authenticated ingress. Keep the legacy API-only
binding available explicitly. Do not enable the new UI on the old all-interface
listener silently. Verify the actual Cloudflare/tunnel routing before cutover.

Browser requests use the operator's same-origin session where ingress provides
one. `API_KEY_ID` and `API_KEY_SECRET` remain server-side upstream credentials;
they are not a browser login mechanism. Never place them in Vite environment
variables, bootstrap payloads, logs returned to the browser, or generated assets.
Validate origins for UI mutations and retain JSON content-type requirements.
Independent account ownership and per-user authorization are a separate product
requirement; the current conversation IDs do not provide that isolation.

Keep `uv`, `.venv`, `pyproject.toml`, and `uv.lock` for Python. Keep npm dependencies
and their lockfile inside `frontend/`. Package the compiled assets with the release
artifact and verify startup from outside the repository working directory.

## What to keep, refactor, migrate and remove

| Area | Action | Intended result |
|---|---|---|
| `src/search/`, search safety markers and normalization | Keep | Existing provider behavior, ephemeral evidence semantics and explicit reranking. |
| Routing, retrieval, reasoning and upstream proxying | Keep; add narrow callable seams where necessary | Shared request behavior for existing clients and UI routes. Avoid a second implementation in TypeScript. |
| `conversation_store.py`, workflow/graph stores and existing data | Keep; add only required transactional operations | Existing IDs, histories, runs and artifacts remain usable. No database replacement. |
| Workflow YAML, graph Python/YAML, `langgraph.json` | Keep | Execution definitions remain backend-owned and independently enabled. |
| `src/llm_metrics.py`, logging and trace collection | Keep | Existing per-call instrumentation remains authoritative. |
| `prompt_state.py`, search request shaping and workflow dispatch helpers | Refactor into backend application functions | Python owns fork policy, evidence construction, thread briefing, param merging and durable transcript writes. |
| Pure metric/cache normalization inside dashboard modules | Extract into a UI-independent Python module | Both Shiny and the browser consume the same normalized values during transition. |
| `proxy.py` composition | Refactor narrowly | One app factory registers every route and lifecycle hook. Currently `create_app()` and the module-level app have different route coverage. |
| `utils.py` config/health/file access | Move behind backend APIs | No browser access to config files, arbitrary paths or server-local URLs. |
| `app_ui.py`, Shiny reactive state, chat rendering, formatters | Rewrite application views in TypeScript; reuse accepted chat components | Browser owns layout, local drafts, view state and rendering. |
| `multi_slot.py` pane state and task controls | Rewrite in TypeScript | Independent request controllers and reducers for up to eight panes. |
| `chat_client.py`, `workflow_client.py`, `graph_client.py` | Replace with typed browser clients | Preserve protocol fixtures and response semantics rather than translating callbacks line by line. |
| Shiny JavaScript DOM handlers, websocket heartbeat/diagnostics, theme plumbing | Remove after cutover | Normal browser events, fetch/SSE lifecycle, CSS theme tokens and request diagnostics. |
| Shiny startup thread and `shiny`, `shinychat`, `shinyswatch` dependencies | Remove after parity and rollback gates | One production backend process and static frontend assets. |

Do not introduce a Node API, microservices, GraphQL, Redis, an external task queue,
or a generic workflow/graph abstraction for this migration. Continue using one
backend worker while execution ownership and locks are process-local.

Suggested file boundaries, created only as their feature is implemented:

```text
frontend/
  src/
    app/                 # routes, application shell, providers, theme
    api/                 # generated types, HTTP client, SSE decoding
    features/
      chat/              # durable chat and turn rendering
      multi-slot/        # ephemeral panes and burst controls
      workflows/         # catalog, params, steps and artifacts
      graphs/            # graph inputs, events and resume controls
      history/           # conversation list and transcript
      traces/            # bounded filtered diagnostics
    components/          # shared metrics, reasoning, evidence, JSON viewer
  tests/                 # browser scenarios and protocol fixtures
src/orchestrator/
  ui_api.py              # additive UI routes and contract models
  chat_turns.py          # extracted single-slot turn preparation/dispatch
  conversation_actions.py # transactional fork operation
  catalog.py             # filtered configuration, health and defaults
src/response_metrics.py   # extracted normalization, no UI imports
```

Keep domain-specific workflow transcript handling beside the workflow code.
Avoid a general frontend plugin registry or a universal page controller.

## API contracts and application behavior

Retain existing chat/search/workflow/graph endpoints for external clients. Add
`/api/ui/v1` only for behavior missing from the existing browser-safe boundary.
The following names and payload outlines are proposed contracts.

| Route | Responsibility |
|---|---|
| `GET /api/ui/v1/bootstrap` | Enabled capabilities, cached model catalog, provider/retrieval/repository choices, defaults, display timezone and UI contract version. No raw config or credentials. |
| `POST /api/ui/v1/catalog/refresh` | Refresh a validated resource selection such as models or retrieval; return catalog values, health status and observation timestamps. Retrieval failures remain visible as status rather than silently rewriting configured choices. |
| `POST /api/ui/v1/chat` | Typed non-streaming direct-chat turn; backend prepares prompt/search/retrieval controls and returns structured text, evidence and diagnostics. |
| `POST /api/ui/v1/chat/stream` | Same command with incremental UI events; reuse existing proxy execution and reasoning handling. |
| `POST /api/ui/v1/conversations/{id}/fork` | Copy durable transcript, apply a new system prompt/reasoning choice, return a new ID and explicit fork reasons. |
| `POST /api/ui/v1/workflow-dispatch` | Create and start a workflow-backed chat turn using the selected controls; return the run ID. Python owns the user/final-answer transcript writes. |
| `GET /api/ui/v1/traces` | Filter by conversation, trace ID and endpoint; enforce a bounded limit and return diagnostic fields. Never accept a filesystem path or expose raw prompt/credential fields. |

Reuse `/models`, `/conversations`, `/conversations/retrieve`,
`/conversations/state`, `/repo-context/repos`, `/workflows`, `/workflow-runs`,
`/graphs`, `/graph-runs`, and their existing actions where they already fit.
Return feature availability explicitly when orchestration subsystems are disabled.
Do not build a second CRUD facade for these resources.

Use typed request/response models at HTTP boundaries without replacing existing
dataclasses or YAML validation. Many current routes accept `Request` and return
untyped dictionaries, so generating TypeScript from today's OpenAPI document alone
would not provide a reliable contract. Type the routes needed by each migration
slice and test schemas against actual responses. OpenAPI also needs explicit SSE
event documentation and fixtures; ordinary JSON response schemas are insufficient.

Direct-chat commands distinguish durable and ephemeral modes. Durable mode sends
only the new turn and a conversation ID; Python loads history. Ephemeral mode
sends completed pane history and omits a conversation ID and explicit slot ID.
The browser submits logical configured endpoint/retrieval identifiers; Python
resolves them to URLs and checks current availability. Keep existing external API
header support while the new UI uses typed body fields.

Backend extraction must preserve these policies:

1. Changing the system prompt or reasoning after a conversation starts creates a
   fork with copied durable history and an explicit notice. An endpoint switch
   follows the existing explicit switch policy. Resolve a fork atomically and
   reject a stale source revision rather than copying half of an active turn.
   Return a conversation revision with state reads and require it for forks.
2. Ad hoc search requests five candidates with inline reranking disabled. Preserve
   query-refiner context, non-blocking search failure, and turn-local evidence
   excluded from durable history. The browser renders structured candidates; it
   does not construct the hidden evidence marker or grounding prompt.
3. Retrieval rewrites the current user turn in Python. Managed citation-suffix
   behavior, if retained, is applied exactly once by backend turn preparation.
4. Workflow dispatch maps controls from the selected spec, ignores disabled
   controls, validates required provider/retrieval/repository choices, and keeps
   uploaded source separate from the submitted request. Preserve visibility of
   intermediate/final/hidden steps and existing output-contract failures.
5. Workflow-backed chat commits its public final answer once from backend
   finalization. Reopening a run or reattaching a viewer cannot append it again.
6. Serialize mutations of the same durable conversation and the same run in the
   current process; unrelated ephemeral panes remain concurrent. A busy conflict
   returns 409 with a useful error. Coordinate run, advance, retry, resume and
   clear actions through the same per-resource guard, including existing API
   entry points. A conversation guard lasts through producer finalization even
   after a browser disconnect. Reject bulk clearing while affected runs are active.

Keep the existing databases and tables. Add a small publication ledger in the
conversation database, uniquely keyed by workflow run ID, conversation ID and
message role, so a transcript append and its receipt commit in one transaction.
Persist the run-to-conversation association before execution; reconcile a
completed run missing its publication after restart. The workflow database can be
configured separately, so do not assume a transaction can span both stores.
Derive a fork revision from durable conversation state under its write lock.
These are narrow additions; no history backfill or general event store is needed.

Include a client-generated request ID for correlation and duplicate detection
while an operation is active. This does not promise durable idempotency across a
process restart. Disable automatic command retries and reconcile history/run
state after an uncertain response. Durable request receipts can be added later if
safe automatic retries become a requirement.

## Streaming, cancellation and recovery

Use streaming `fetch` for POST requests and a proper SSE parser. Native
[`EventSource` exposes a URL and credential option](https://developer.mozilla.org/en-US/docs/Web/API/EventSource/EventSource),
which does not express these JSON POST commands. Handle split UTF-8 characters,
partial frames, multiple frames per read, CRLF, comments/heartbeats, terminal
markers and errors after the HTTP response has started.

Keep existing OpenAI SSE and workflow/graph event formats intact for external
clients. The new chat route can adapt them into a documented discriminated event
union: `turn_started`, `search_results`, `reasoning_delta`, `content_delta`,
`metrics`, `turn_completed`, and `error`. A started event carries the effective
conversation ID and fork notice. When raw JSON inspection is requested, include
the original normalized completion payload or `raw_frame` events so unknown
upstream fields remain inspectable. Stream events contain raw structured values;
rendering tags and footer Markdown never enter model history.

Share byte decoding and framing across features. Chat, workflows and graphs each
have their own typed event reducer. Do not force their state machines into one
universal run model. Errors before stream creation use a typed JSON error; errors
afterward use a terminal stream event. Unexpected EOF yields an interrupted state,
not a successful completion. Never retry a generation because parsing failed.

The chat producer already survives downstream disconnect and finalizes history.
Preserve that ownership when introducing a UI adapter: closing the adapter's
subscription must reach the proxy consumer cleanup without cancelling the
producer or leaving an accumulating subscriber queue.
The existing proxy queue is unbounded; include it in the bounded-subscriber work
and test slow readers as well as fully disconnected readers.

Workflow/graph streams currently drive execution directly from their response
generators. Before migrating those execution views, separate execution ownership
from observation using tracked tasks in the existing Python process. Keep a
bounded live-event subscription per viewer; snapshots and existing persisted
artifacts/events remain authoritative. Reuse current command routes and add
passive per-domain observation routes only where needed. Attaching a viewer must
never start a second execution. A slow viewer can be disconnected and told to
reload the snapshot; it must not block model execution or grow memory indefinitely.

This task ownership is process-local, not crash recovery. On restart, reconcile
stale running records using existing failed/error fields with an explicit
interruption reason; the UI can label that outcome Interrupted without adding a
new persisted status. Preserve graph resume only where graph checkpoint behavior
supports it. Do not promise replay of token deltas, cursor-based reconnection, or
automatic resumption without corresponding persistence support.

Distinguish **Stop receiving** from an acknowledged backend cancellation. Existing
chat inference may continue after the browser aborts. Preserve that behavior and
show it clearly. Add a true cancellation API only if upstream cancellation and
history finalization can be implemented consistently. A workflow or graph without
that support offers detaching from the live view, with execution status obtained
from a snapshot. Reconnection fetches state; it never repeats an execution POST.

Use SSE comment heartbeats while a stream is idle and verify that ingress does
not buffer them. Replace the Shiny websocket heartbeat with request/stream
diagnostics and trace IDs; the two mechanisms protect different connections.

## Browser state, navigation and diagnostics

| State | Owner and lifetime |
|---|---|
| Durable messages, route/reasoning pin, workflow/graph status and artifacts | Backend and existing stores. Reload from snapshots. |
| Endpoint/provider/retrieval catalogs and run lists | Query cache; refresh/invalidate deliberately. |
| Current streaming text, reasoning, immutable request settings and response metadata | Feature reducer, keyed by request ID; one abort controller per request. |
| Multi-slot transcripts and drafts | Browser memory for the current application session. Hidden panes retain state; reset/endpoint changes clear it. |
| Theme, metrics visibility and other harmless preferences | Local storage. Avoid persisting prompts, attachments, service tokens or session transcripts there. |
| Selected conversation/run, trace filters and active page | URL path/search parameters where useful for reload and links. |

Use `/ui/chat`, `/ui/multi-slot`, `/ui/workflows/:runId?`,
`/ui/graphs/:runId?`, `/ui/history/:conversationId?`, and `/ui/traces`.
The shell supplies navigation and catalog status. Each feature owns its controls;
generation settings and diagnostics use shared components. Preserve reasoning,
raw JSON inspection, file augmentation, themes and logout. Keep forms usable by
keyboard and expose busy, failed and interrupted states accessibly.

Multi-slot requests snapshot settings at submission, skip busy panes during a
burst, preserve individual drafts, and exclude failed/cancelled replies from the
next request history. Preserve the eight-pane limit, hidden-pane histories and
disabled destructive controls while panes are active. Pane number remains a UI
identity, not a physical inference slot. Route changes must not unmount an active
request accidentally: keep active pane/stream state above route views for the
application session; a full browser reload ends ephemeral session state.

Read existing UTF-8 source uploads in the browser; submit text, filename and size
metadata to the turn/run command. Enforce encoding, aggregate byte and request
limits on both sides; never trust a supplied local path. Preserve current
augmentation semantics before considering binary document ingestion or durable
attachment storage.

Extract metric/cache normalization from the dashboard into Python and reuse its
current tests. Expose normalized numbers with explicit units and nullable missing
values. Frontend formatting must preserve zero, distinguish missing from zero,
avoid division by zero, and never infer unreported tokens or throughput. Freeze
the model catalog/configuration associated with each response; refreshing models
must not rewrite historical footers. Keep per-call token/rate/TTFT values distinct
from workflow total elapsed time. Browser-observed timing uses a monotonic browser
clock and is labeled separately from backend-observed timing; compare durations,
never subtract timestamps from different clocks. Store wall timestamps in UTC
and apply the configured display timezone at rendering.

History reload currently cannot reconstruct every chat metrics footer because
durable messages exclude telemetry. Preserve session footers and show unavailable
diagnostics honestly after reload. A persisted per-turn diagnostics table is an
optional later feature, not a prerequisite for this migration.

## Delivery sequence and acceptance gates

Ship each slice as a reviewable change. Keep the old UI callable until the final
gate. Estimates are rough engineer-days for one maintainer familiar with the
repository, including implementation and tests; validate them after the first
working stream. They are sequencing guidance, not a delivery commitment.

| Phase | Scope | Exit condition | Effort |
|---|---|---|---|
| 0. Capture behavior and evaluate reuse | Record sanitized chat SSE/JSON, search, workflow/graph snapshots and error fixtures; inventory dashboard regressions and ingress behavior. Correct header/health discrepancies. Run the assistant-ui/LibreChat/big-AGI reuse evaluation above. | Written parity checklist, deterministic fixtures, documented reuse decision and selected frontend stack. | 3–5 days |
| 1. Complete backend boundary | Consolidate app composition; extract catalog, metric normalization, prompt/search preparation and fork operation; type the APIs needed for chat. Keep Shiny using extracted functions. | Existing behavior passes Python tests; new routes work without importing Shiny; catalog and responses contain no service secrets. | 3–5 days |
| 2. Prove one complete browser flow | Scaffold frontend, selected chat-component integration, shell, typed client, streaming decoder and `/ui` serving; implement durable direct chat with reasoning, errors, metrics, forks and history reload. | Browser sends two turns, changes controls to fork, renders streamed and JSON replies, then reloads history. Legacy clients still work. | 3–5 days |
| 3. Chat parity | Add ad hoc search, retrieval controls, uploads, raw JSON and multi-slot/burst behavior. | Search evidence remains ephemeral; eight concurrent panes remain isolated; stop/failure/settings behavior matches the checklist. | 3–5 days |
| 4. Execution views | Extract workflow dispatch/transcript finalization; introduce tracked workflow/graph task ownership and observation; migrate run creation, advance/run, retry, artifacts and supported graph resume. | Leaving/reopening a view neither repeats execution nor loses the durable final transcript; workflows and graphs retain separate APIs. | 5–8 days |
| 5. Operational parity and cutover | Finish traces, timezone/theme/logout behavior, ingress smoke checks, asset packaging, docs and entry point; switch the default UI. | Full parity checklist passes behind the real ingress; rollback path exercised; no Shiny-only product feature remains. | 2–4 days |
| 6. Remove legacy | Delete unused dashboard code and Shiny dependencies, refresh `uv.lock`, remove obsolete tests and launch option after an agreed validation window. | Production starts with compiled assets and Python only; retained regression tests cover moved policies. | 1–2 days |

Expected total for the component-reuse/custom-console route: approximately
**20–34 engineer-days**, including the added reuse evaluation. No implementation
savings from an upstream frontend are assumed until the prototype demonstrates
them. Full-product adoption requires its own estimate. The largest uncertainty
is execution disconnect/finalization behavior, followed by undocumented deployment
assumptions. Do not replace the full UI before proving phase 2; that slice tests
the architecture with a useful end-to-end feature.

## Validation and rollback

Preserve existing Python backend tests. Move dashboard policy/normalization tests
with the extracted code and port browser-specific behavior tests; delete obsolete
Shiny plumbing tests only when their feature is replaced. Use fixtures as reference
comparisons so Python and TypeScript do not drift on response semantics.

Required regression scenarios:

- Durable chat sends new turns only; full-history replay remains rejected. Prompt
  and reasoning changes fork; endpoint changes retain the explicit switch policy.
- Search refinement/degradation and retrieval results preserve prompt/history
  boundaries. Workflow output visibility and final transcript persistence survive
  disconnect. Disabled subsystems render as unavailable capabilities.
- Streams handle chunk boundaries, split Unicode, usage-only frames, reasoning,
  proxy errors, abrupt EOF and abort. A second viewer cannot execute a run again.
- Eight panes maintain independent histories and settings snapshots. Bursts skip
  busy panes; failed/cancelled output is excluded; zero temperature is submitted
  and blank generation controls preserve upstream defaults.
- Missing/zero metrics, rate denominators, cache telemetry and per-call workflow
  metrics agree with current reference cases. Model refresh leaves earlier
  diagnostics unchanged; timezone formatting does not alter stored timestamps.
- Packaged UI deep links load; missing assets and unknown APIs return correct
  errors; endpoint names cannot collide with UI routes. Credentials are absent
  from browser bundles and responses. Real-ingress streams flush and remain live.
- Reload/reconnect reads state without automatic mutation replay; a backend
  restart reports interrupted work truthfully rather than leaving permanent busy
  indicators. Concurrent fork/run/retry/clear conflicts are deterministic.

Run Python checks with `uv run --locked pytest`. Once the frontend exists, its
checks are `npm ci`, type checking, unit tests, production build, and deterministic
browser tests. Live model/provider/ingress smoke tests supplement these checks;
they should not be the sole correctness evidence. Establish a measured baseline
for eight-pane rendering and stream delivery before setting performance budgets.

Additive APIs and optional transactional/schema additions must remain readable by
the legacy UI throughout coexistence. Keep the previous asset build and launcher
configuration for rollback; leave existing IDs/data in place. Do not roll back to
older backend code after new persisted statuses/schema changes without verifying
compatibility. Exercise rollback before removing Shiny, then remove only code
proven unused and update the architecture, API, dashboard and setup guides.

Decisions to verify during phase 0 are maintainer framework familiarity, the actual
remote access boundary, acceptable behavior after closing a run view, and whether
session-only multi-slot history remains sufficient. The recommendation proceeds
with the assumptions above; none requires a backend language rewrite.
