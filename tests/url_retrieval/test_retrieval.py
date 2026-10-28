import asyncio
import gzip
from pathlib import Path

import httpx
import pytest

from src import url_retrieval as retrieval
from src.url_retrieval import URLRetrievalConfig, extract_urls, fetch_url_output


def test_discovery_preserves_encoded_paths_queries_and_balanced_delimiters():
    assert extract_urls(
        '[docs](HTTPS://EXAMPLE.COM/a%2Fb_(v1)?x=%2f#part), '
        '<https://example.com/a%2Fb_(v1)?x=%2f#other>; '
        '`https://other.example/` "http://other.example:80" '
        'https://other.example https:// https://[invalid]'
    ) == [
        "https://example.com/a%2Fb_(v1)?x=%2f",
        "https://other.example/",
        "http://other.example",
    ]


def test_discovery_includes_quoted_code_urls_but_not_schemeless_addresses():
    assert extract_urls(
        "'https://example.org/foo'. ```https://code.example/bar``` "
        "example.com www.example.com ftp://example.com https://example.com/?q=O'Reilly"
    ) == [
        "https://example.org/foo", "https://code.example/bar", "https://example.com/?q=O'Reilly",
    ]
    assert extract_urls("") == []


@pytest.mark.parametrize("options", [
    {"concurrency": 0}, {"timeout_seconds": 0}, {"max_body_bytes": 0},
    {"max_text_chars": 0}, {"max_redirects": -1},
])
def test_invalid_limits_are_rejected_before_running(options):
    with pytest.raises(ValueError):
        URLRetrievalConfig(**options)


@pytest.mark.asyncio
async def test_empty_batch_performs_no_network_activity(monkeypatch):
    def unexpected_transport():
        pytest.fail("empty input must not open a transport")

    monkeypatch.setattr(retrieval, "PublicHTTPTransport", unexpected_transport)
    output = await fetch_url_output([])
    assert output["json"] == {"results": []}
    assert output["metadata"] == {
        "kind": "url_fetch", "attempted": 0, "successful": 0, "failed": 0,
    }


@pytest.mark.asyncio
async def test_html_extraction_uses_real_trafilatura_and_preserves_tables(web_backend):
    html = (Path(__file__).parent / "fixtures/article.html").read_text()
    requests, _ = web_backend(lambda request: httpx.Response(
        200, text=html, headers={"content-type": "text/html; charset=utf-8"},
    ))
    output = await fetch_url_output(["https://docs.example/article"])
    result = output["json"]["results"][0]
    assert result["status"] == "success"
    assert result["title"] == "Public Reference"
    assert result["extraction_method"] == "trafilatura"
    assert "retrieval pipeline" in result["text"]
    assert "Concurrency" in result["text"]
    assert "Navigation menu" not in result["text"]
    assert "secret_script" not in result["text"]
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "docs.example"
    assert requests[0].extensions["sni_hostname"] == "docs.example"
    assert "authorization" not in requests[0].headers
    assert "untrusted external material" in output["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("extraction", [None, " \n", RuntimeError("extractor failed")])
async def test_html_falls_back_to_cleaned_text(web_backend, monkeypatch, extraction):
    def extract(*args, **kwargs):
        if isinstance(extraction, Exception):
            raise extraction
        return extraction

    monkeypatch.setattr(retrieval.trafilatura, "extract", extract)
    web_backend(lambda request: httpx.Response(
        200, headers={"content-type": "application/xhtml+xml"},
        text="<head><title>Example</title><style>bad_css</style></head>"
             "<nav>bad_nav</nav><main><h1>Heading</h1><p>Readable content</p></main>"
             "<script>bad_script</script><footer>bad_footer</footer>",
    ))
    result = (await fetch_url_output(["https://example.com"]))["json"]["results"][0]
    assert result["status"] == "success"
    assert result["title"] == "Example"
    assert result["extraction_method"] == "beautifulsoup"
    assert result["text"] == "Heading\nReadable content"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type,text", [
    ("text/plain", "Plain text"),
    ("application/json", '{"key": "value"}'),
    ("application/problem+json", '{"detail": "example"}'),
])
async def test_textual_content_is_returned_without_html_extraction(web_backend, content_type, text):
    web_backend(lambda request: httpx.Response(
        200, text=text, headers={"content-type": content_type},
    ))
    result = (await fetch_url_output(["https://example.com"]))["json"]["results"][0]
    assert result["text"] == text
    assert result["extraction_method"] == "plain_text"


@pytest.mark.asyncio
@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.0.0.1", "169.254.169.254", "[::1]", "[::ffff:127.0.0.1]",
    "224.0.0.1", "240.0.0.1", "localhost", "mixed.example",
])
async def test_nonpublic_targets_are_blocked_before_connecting(web_backend, host):
    requests, _ = web_backend(
        lambda request: pytest.fail("blocked host must not connect"),
        dns={"localhost": ["127.0.0.1"], "mixed.example": ["93.184.216.34", "10.0.0.1"]},
    )
    result = (await fetch_url_output([f"http://{host}/"]))["json"]["results"][0]
    assert result["status"] == "error"
    assert "nonpublic" in result["error"]
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", [
    "http://127.0.0.1/private", "http://user:password@example.com/", "ftp://example.com/",
])
async def test_redirect_targets_are_revalidated(web_backend, destination):
    requests, _ = web_backend(lambda request: httpx.Response(
        302, headers={"location": destination},
    ))
    result = (await fetch_url_output(["https://example.com"]))["json"]["results"][0]
    assert result["status"] == "error"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_public_redirect_preserves_https_identity_and_encoded_target(web_backend):
    def handler(request):
        if request.headers["host"] == "first.example":
            return httpx.Response(302, headers={"location": "https://second.example/a%2Fb?q=%2f#top"})
        return httpx.Response(200, text="Redirected", headers={"content-type": "text/plain"})

    requests, resolved = web_backend(handler)
    result = (await fetch_url_output(["https://first.example"]))["json"]["results"][0]
    assert result["final_url"] == "https://second.example/a%2Fb?q=%2f"
    assert result["text"] == "Redirected"
    assert resolved == ["first.example", "second.example"]
    assert requests[1].headers["host"] == "second.example"
    assert requests[1].extensions["sni_hostname"] == "second.example"


@pytest.mark.asyncio
@pytest.mark.parametrize("response,error", [
    (httpx.Response(404), "404"),
    (httpx.Response(500), "500"),
    (httpx.Response(302), "Location"),
    (httpx.Response(200, content=b"pdf", headers={"content-type": "application/pdf"}), "Unsupported"),
    (httpx.Response(200, content=b"missing type"), "missing"),
    (httpx.Response(200, text="", headers={"content-type": "text/plain"}), "no readable"),
    (httpx.Response(200, text="<script>ignored</script>", headers={"content-type": "text/html"}), "no readable"),
    (httpx.Response(200, text="<![invalid]>", headers={"content-type": "text/html"}), "no readable"),
])
async def test_document_failures_are_explicit_results(web_backend, response, error):
    web_backend(lambda request: response)
    output = await fetch_url_output(["https://example.com"])
    result = output["json"]["results"][0]
    assert result["status"] == "error"
    assert error in result["error"]
    assert result["text"] == ""
    assert output["metadata"]["failed"] == 1


@pytest.mark.asyncio
async def test_limits_apply_to_redirects_decompressed_body_and_extracted_text(web_backend):
    def handler(request):
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/redirect"})
        return httpx.Response(
            200, content=gzip.compress(b"x" * 40),
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
        )

    requests, _ = web_backend(handler)
    output = await fetch_url_output(
        ["https://example.com/redirect", "https://example.com/body"],
        config=URLRetrievalConfig(max_redirects=1, max_body_bytes=39),
    )
    assert "Redirect limit" in output["json"]["results"][0]["error"]
    assert "exceeded 39 bytes" in output["json"]["results"][1]["error"]
    assert len(requests) == 3
    result = (await fetch_url_output(
        ["https://example.com/body"], config=URLRetrievalConfig(max_text_chars=10),
    ))["json"]["results"][0]
    assert result["text"] == "x" * 10
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_total_deadline_includes_dns_resolution(web_backend, monkeypatch):
    requests, _ = web_backend(lambda request: pytest.fail("DNS should time out"))

    async def slow_dns(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", slow_dns)
    result = (await fetch_url_output(
        ["https://example.com"], config=URLRetrievalConfig(timeout_seconds=0.01),
    ))["json"]["results"][0]
    assert result["error"] == "TimeoutError"
    assert requests == []


@pytest.mark.asyncio
async def test_total_deadline_stops_a_streaming_body_and_closes_it(web_backend):
    closed = asyncio.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.004)
                yield b"x"

        async def aclose(self):
            closed.set()

    web_backend(lambda request: httpx.Response(
        200, stream=SlowBody(), headers={"content-type": "text/plain"},
    ))
    result = (await fetch_url_output(
        ["https://example.com"], config=URLRetrievalConfig(timeout_seconds=0.01),
    ))["json"]["results"][0]
    assert result["error"] == "TimeoutError"
    assert closed.is_set()


@pytest.mark.asyncio
async def test_unexpected_failure_cancels_other_requests(web_backend):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        if request.url.path == "/bug":
            await started.wait()
            raise RuntimeError("implementation bug")
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    web_backend(handler)
    with pytest.raises(RuntimeError, match="implementation bug"):
        await fetch_url_output(["https://example.com/bug", "https://example.com/slow"])
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_concurrency_ordering_partial_failures_and_no_automatic_retry(web_backend):
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001 * (8 - int(request.url.path[1:])))
        active -= 1
        if request.url.path == "/2":
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(200, text=request.url.path, headers={"content-type": "text/plain"})

    requests, _ = web_backend(handler)
    urls = [f"https://example.com/{i}" for i in range(8)]
    output = await fetch_url_output(urls)
    assert [result["url"] for result in output["json"]["results"]] == urls
    assert output["metadata"] == {"kind": "url_fetch", "attempted": 8, "successful": 7, "failed": 1}
    assert len(requests) == 8
    assert peak == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("urls", [None, "https://example.com", [1]])
async def test_invalid_batch_input_fails_at_the_boundary(urls):
    with pytest.raises(ValueError, match="JSON array"):
        await fetch_url_output(urls)
