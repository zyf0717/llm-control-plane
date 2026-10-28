# URL Retrieval

`url_retrieval` is a standalone YAML workflow and LangGraph graph. Both run the
same deterministic pipeline: extract explicit HTTP/S URLs, fetch their contents,
and return readable source documents. No LLM, search provider, or Retrieval
endpoint is used. Selecting this workflow or graph is explicit; ordinary chat
and other shipped pipelines do not automatically fetch URLs.

## Invocation

Create a workflow run with `POST /workflows/url_retrieval/runs`:

```json
{
  "params": {
    "latest_user_prompt": "Read https://example.com/article and https://example.org/docs"
  }
}
```

Then call `POST /workflow-runs/{run_id}/run` or advance the two steps separately
with `POST /workflow-runs/{run_id}/advance`. The `/run-stream` endpoint emits the
usual workflow events. Retry `fetch_urls` to repeat retrieval without repeating
URL discovery.

For the graph, create a run with `POST /graphs/url_retrieval/runs`:

```json
{
  "input": {
    "latest_user_prompt": "Read https://example.com/article"
  }
}
```

Then use `POST /graph-runs/{run_id}/run` or `/stream`. Neither run needs an LLM
endpoint. Both are also listed in the existing dashboard selectors.

## Discovery and extraction

Discovery includes bare HTTP/S URLs, Markdown destinations, angle-bracket links,
and URLs inside code or quoted text. Surrounding punctuation and unmatched
closing delimiters are removed. Duplicates are fetched once, in first-occurrence
order; fragments are ignored, while encoded paths and queries are preserved.
Bare domains and non-HTTP schemes are not discovered. Retrieved links are not
crawled. A prompt with no URLs produces an empty successful result without
opening a network client.

HTML/XHTML is processed by Trafilatura with comments excluded and tables
included. If Trafilatura raises or produces empty/whitespace-only content,
BeautifulSoup extracts readable text after removing scripts, styles, head,
navigation, footer, and other non-content elements. Raw HTML is not included.
Text MIME types and JSON are returned as text. PDFs, binary files, and responses
without a supported Content-Type produce explicit document errors. JavaScript
is not executed.

## Results and limits

The first workflow step stores `url_candidates`, whose `json.urls` contains the
discovered URLs. The fetch step stores `url_documents` with the standard output
envelope:

- `json.results`: ordered document records with `url`, `final_url`, `status`
  (`success` or `error`), `http_status`, `content_type`, `title`, `text`,
  `extraction_method` (`trafilatura`, `beautifulsoup`, or `plain_text`), UTC
  `fetched_at`, `truncated`, and `error`. Unavailable optional metadata is null;
  failed documents have empty text.
- `text`: readable source text and per-document errors, labeled as untrusted
  external material. The workflow saves this as an artifact and final visible
  step output.
- `metadata`: `kind`, `attempted`, `successful`, and `failed` counts.

The graph uses the same `outputs.url_candidates` and `outputs.url_documents`
envelopes and sets `final` to the rendered text. Graph streaming follows the
existing event format: the `fetch_urls` update contains the results, and the
completed snapshot records the last event.

Defaults are four concurrent documents, a ten-second total network deadline per
document (including DNS and redirects), five redirects, 2.5 MB of decompressed
response bytes, and 30,000 extracted characters per document. Excess response
bytes fail retrieval; excess extracted text is truncated and marked. There is
no URL-count cap, automatic retry, or persistent content cache. Shared Python
callers can supply `URLRetrievalConfig` to adjust these limits.

Only public HTTP/S destinations are allowed. DNS answers and every redirect are
validated, and connections are pinned to a validated IP while retaining the
original Host and TLS hostname. Embedded credentials, nonpublic addresses,
environment proxies, and forwarding application authentication are unsupported.

Expected failures remain per-document results, including when all URLs fail.
Unexpected implementation errors fail the workflow step or graph run. Existing
run inspection and retry mechanisms apply.

## Reuse

Custom YAML workflows can use `kind: url_extract` with a text prompt, followed
by `kind: url_fetch` with a rendered JSON URL array, as shown in the shipped
workflow. Python graphs can call `extract_url_output` and `fetch_url_output`
from `src.url_retrieval` directly. The graph and workflow engines remain
independent; nested workflow execution is not introduced.
