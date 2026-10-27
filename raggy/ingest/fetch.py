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
from pathlib import Path, PurePosixPath
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


MEDIA_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "soundcloud.com", "twitch.tv", "dailymotion.com")


def is_media_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in MEDIA_HOSTS)


def _get_text(url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=20.0) as c:
        r = c.get(url, headers={"user-agent": _user_agent()})
        if r.status_code >= 400:
            raise FetchError(f"{url}: HTTP {r.status_code}")
        return r.text


def fetch_media(url: str, options: LoadOptions | None = None) -> Document:
    """A video or podcast page (week 11), through yt-dlp: its published English captions
    if it has any (no model needed), else its audio, transcribed like a local file."""
    try:
        import yt_dlp
    except ImportError:
        raise LoadError(f"{url}: media pages need yt-dlp: run `uv sync --extra media`") from None
    if yt_dlp is None:
        raise LoadError(f"{url}: media pages need yt-dlp: run `uv sync --extra media`")
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        tracks = info.get("subtitles") or {}
        lang = next((k for k in tracks if k == "en" or k.startswith("en-") or k.startswith("en_")), None)
        vtt = next((t for t in tracks.get(lang, []) if t.get("ext") == "vtt"), None) if lang else None
        if vtt is not None:
            doc = load_bytes(_get_text(vtt["url"]).encode(), url, "subtitles", options)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                opts = {"quiet": True, "no_warnings": True, "format": "bestaudio[ext=m4a]/bestaudio",
                        "outtmpl": f"{tmp}/audio.%(ext)s"}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.extract_info(url, download=True)
                audio = next(Path(tmp).glob("audio.*"))
                doc = load_bytes(audio.read_bytes(), url, "audio", options)
    except LoadError:
        raise
    except Exception as e:  # yt-dlp raises its own DownloadError / ExtractorError for pages it can't handle
        raise LoadError(f"{url}: yt-dlp couldn't read it ({type(e).__name__}: {e})") from e
    doc.metadata.update(kind="url", final_url=info.get("webpage_url") or url)
    if info.get("title"):
        doc.metadata["title"] = info["title"]
    return doc


def fetch_url(url: str, options: LoadOptions | None = None, client: httpx.Client | None = None,
              max_bytes: int = 20_000_000) -> Document:
    if is_media_url(url):
        return fetch_media(url, options)
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
