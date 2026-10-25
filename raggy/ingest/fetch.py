"""Fetch one web page (or a PDF or text file at a URL) and load it like a file.

The response's Content-Type picks the loader, because a URL's path often
says nothing ("/download?id=7"). Only when the server sends no useful type
does the path's suffix decide. Bodies are read in pieces and refused past a
size limit, so a link to a 4 GB video can't fill memory.

raggy fetches the URLs you give it; it doesn't follow links or crawl.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx

from raggy.ingest.loaders import SUFFIXES, LoadError, LoadOptions, load_bytes
from raggy.types import Document

CONTENT_TYPES = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}
GENERIC_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


class FetchError(LoadError):
    """A URL couldn't be fetched, or returned something raggy can't read."""


def is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https") and bool(urlparse(s).netloc)


def _user_agent() -> str:
    try:
        v = version("raggy")
    except PackageNotFoundError:
        v = "dev"
    return f"raggy/{v} (+https://github.com/krishankant/raggy)"


def _kind(content_type: str, url: str) -> str:
    mime = content_type.split(";")[0].strip().lower()
    if mime in CONTENT_TYPES:
        return CONTENT_TYPES[mime]
    if mime in GENERIC_TYPES:
        kind = SUFFIXES.get(PurePosixPath(urlparse(url).path).suffix.lower())
        if kind:
            return kind
    raise FetchError(f"{url}: unsupported content type {mime!r}")


def fetch_url(url: str, options: LoadOptions | None = None, client: httpx.Client | None = None,
              max_bytes: int = 20_000_000) -> Document:
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=20.0)
    try:
        with client.stream("GET", url, headers={"user-agent": _user_agent()}) as resp:
            if resp.status_code >= 400:
                raise FetchError(f"{url}: HTTP {resp.status_code}")
            kind = _kind(resp.headers.get("content-type", ""), str(resp.url))
            body = bytearray()
            for piece in resp.iter_bytes():
                body += piece
                if len(body) > max_bytes:
                    raise FetchError(f"{url}: larger than {max_bytes:,} bytes")
            final_url, last_modified = str(resp.url), resp.headers.get("last-modified")
    except httpx.HTTPError as e:
        raise FetchError(f"{url}: {e}") from e
    finally:
        if own:
            client.close()
    doc = load_bytes(bytes(body), url, kind, options)
    doc.metadata.update(kind="url", final_url=final_url)
    if last_modified:
        try:
            doc.metadata["modified"] = parsedate_to_datetime(last_modified).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            pass
    return doc
