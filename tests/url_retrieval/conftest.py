import asyncio
import inspect
import ipaddress
import socket

import httpx
import pytest

from src import url_retrieval


@pytest.fixture
def web_backend(monkeypatch):
    """Mock the wire after real URL validation and IP pinning."""
    def install(handler, *, dns=None):
        requests = []
        resolved = []

        async def getaddrinfo(host, port, **kwargs):
            resolved.append(host)
            try:
                addresses = [str(ipaddress.ip_address(host))]
            except ValueError:
                addresses = (dns or {}).get(host, ["93.184.216.34"])
            return [
                (socket.AF_INET6 if ":" in address else socket.AF_INET,
                 socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))
                for address in addresses
            ]

        async def dispatch(request):
            requests.append(request)
            response = handler(request)
            return await response if inspect.isawaitable(response) else response

        def transport(**kwargs):
            assert kwargs["limits"].max_keepalive_connections == 0
            return httpx.MockTransport(dispatch)

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
        monkeypatch.setattr(url_retrieval.httpx, "AsyncHTTPTransport", transport)
        return requests, resolved

    return install
