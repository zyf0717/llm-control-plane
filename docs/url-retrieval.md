# URL Retrieval

`url_retrieval` is a standalone YAML workflow and LangGraph graph. Both extract
explicit HTTP/S URLs and fetch their contents deterministically, then use the
selected model to answer the original prompt. The answer follows the requested
instructions and output format, with the retrieved pages as source material.
No LLM is involved in discovery, fetching, or parsing. No search provider or
Retrieval endpoint is used. Selecting this workflow or graph is explicit;
ordinary chat and other shipped pipelines do not automatically fetch URLs.

## Invocation

Create a workflow run with `POST /workflows/url_retrieval/runs`:

```json
{
  "params": {
    "latest_user_prompt": "Read https://example.com/article and https://example.org/docs"
  },
  "endpoint": "primary"
}
```

Then call `POST /workflow-runs/{run_id}/run` or advance the three steps separately
with `POST /workflow-runs/{run_id}/advance`. The `/run-stream` endpoint emits the
usual workflow events. Retry `fetch_urls` to repeat retrieval without repeating
URL discovery; the dependent answer step is also rerun.

For the graph, create a run with `POST /graphs/url_retrieval/runs`:

```json
{
  "input": {
    "latest_user_prompt": "Read https://example.com/article"
  },
  "config": {"configurable": {"endpoint": "primary"}}
}
```

Then use `POST /graph-runs/{run_id}/run` or `/stream`. The workflow requires a
concrete model endpoint for its final answer; the graph uses its configured
endpoint (default `smart`). Both are listed in the existing dashboard selectors.
Conversation history, supplied instructions, and uploaded context are passed to
the final model step using the same context handling as the other workflows.

## Discovery and extraction

Discovery includes bare HTTP/S URLs, Markdown destinations, angle-bracket links,
and URLs inside code or quoted text. Surrounding punctuation and unmatched
closing delimiters are removed. Duplicates are fetched once, in first-occurrence
order; fragments are ignored, while encoded paths and queries are preserved.
Bare domains and non-HTTP schemes are not discovered. Retrieved links are not
crawled. A prompt with no URLs skips network retrieval and still receives an
answer from the model using the available context.

BeautifulSoup extracts readable text from HTML/XHTML after removing scripts,
styles, head, navigation, footer, and other non-content elements. Headings,
link text, and table text are retained. Raw HTML is not included.
Each HTML document also includes `metadata.links`, a list of `{text, url}`
pairs in page order. Relative destinations use the final fetched URL and any
HTML `<base href>`. HTTP/S links from navigation and footer are included in
metadata even though those sections are omitted from the body text. Identical
label/destination pairs are deduplicated; distinct labels and section fragments
are retained. Empty, non-HTTP/S, malformed, and credential-bearing destinations
are excluded. Discovered destinations are not fetched.
Text MIME types and JSON are returned as text. PDFs, binary files, and responses
without a supported Content-Type produce explicit document errors. JavaScript
is not executed.

## Results and limits

The first workflow step stores `url_candidates`, whose `json.urls` contains the
discovered URLs. The fetch step stores `url_documents` with the standard output
envelope:

- `json.results`: ordered document records with `url`, `final_url`, `status`
  (`success` or `error`), `http_status`, `content_type`, `title`, `text`,
  `extraction_method` (`beautifulsoup` or `plain_text`), UTC
  `fetched_at`, `truncated`, and `error`. Unavailable optional metadata is null;
  failed documents have empty text.
- Per-document `metadata`: HTML results include `links` and `links_truncated`;
  non-HTML results have an empty metadata object. Link discovery is independent
  of body-text truncation, so links beyond the retained text can still appear.
- `text`: readable source text and per-document errors, labeled as untrusted
  external material. The workflow saves this as an artifact, while the final
  chat response comes from the subsequent answer step.
- `metadata`: `kind`, `attempted`, `successful`, and `failed` counts.

The rendered source text includes discovered links for the answer step, with a
separate marker when link metadata is truncated.

Both engines store the model answer in `outputs.reply`. The graph uses the same
`outputs.url_candidates` and `outputs.url_documents` envelopes and sets `final`
to the model's answer. Graph streaming follows the existing event format:
`fetch_urls` contains source documents, `answer_from_urls` contains the final
answer, and the completed snapshot records the last event.

Defaults are four concurrent documents, a ten-second total network deadline per
document (including DNS and redirects), five redirects, 2.5 MB of decompressed
response bytes, and 30,000 extracted characters per document. Excess response
bytes fail retrieval; excess extracted text is truncated and marked. There is
also a separate link-metadata budget of 100 unique label/destination pairs and
30,000 characters across their labels and URLs per page. The character ceiling
uses `max_text_chars`; the count ceiling uses `max_links`. Discovery stops at
either ceiling and sets `metadata.links_truncated` when another eligible link
cannot fit. These limits apply independently of the body-text limit. There is
no URL-count cap, automatic retry, or persistent content cache. Shared Python
callers can supply `URLRetrievalConfig` to adjust these limits.

Only public HTTP/S destinations are allowed. DNS answers and every redirect are
validated, and connections are pinned to a validated IP while retaining the
original Host and TLS hostname. Embedded credentials, nonpublic addresses,
environment proxies, and forwarding application authentication are unsupported.

Expected failures remain per-document results, including when all URLs fail;
the answer step receives those failures and is instructed to state material
limitations without inventing page contents.
Unexpected implementation errors fail the workflow step or graph run. Existing
run inspection and retry mechanisms apply.

## Reuse

Custom YAML workflows can use `kind: url_extract` with a text prompt, followed
by `kind: url_fetch` with a rendered JSON URL array, as shown in the shipped
workflow. Python graphs can call `extract_url_output` and `fetch_url_output`
from `src.url_retrieval` directly. These operations remain model-free and can
be consumed without running the final answer step. The graph and workflow
engines remain independent; nested workflow execution is not introduced.
