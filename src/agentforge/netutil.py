"""Hardened outbound HTTP helpers (TLS verified, no redirects, size/time limits)."""
from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class NetworkError(Exception):
    """Generic network failure; details are never exposed to callers."""


def validate_provider_url(url: str) -> str:
    """Providers must use https; plain http is only allowed for loopback (e.g. local Ollama)."""
    parsed = urlparse(url)
    if parsed.hostname and (
        parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS)
    ):
        return url.rstrip("/")
    raise ValueError("base_url must use https (http is allowed for loopback hosts only)")


def assert_public_https(url: str) -> None:
    """SSRF guard for the fetch tool: https only, every resolved address must be public."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise NetworkError("only https URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise NetworkError("host could not be resolved") from exc
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise NetworkError("target address is not public")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def send(request: urllib.request.Request, timeout: float, max_bytes: int) -> bytes:
    try:
        with _OPENER.open(request, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise NetworkError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NetworkError("request failed") from exc
    if len(data) > max_bytes:
        raise NetworkError("response too large")
    return data
