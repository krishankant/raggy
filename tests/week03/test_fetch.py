import httpx
import pytest

from raggy.ingest.fetch import FetchError, fetch_url, is_url
from raggy.ingest.loaders import LoadError

from .conftest import PAGE, make_pdf


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_is_url():
    assert is_url("https://example.com/a") and is_url("http://x.org")
    assert not is_url("docs/a.md") and not is_url("ftp://x.org/f") and not is_url("file:///etc/passwd")


def test_fetch_html_main_content():
    def handler(req):
        assert req.headers["user-agent"].startswith("raggy/")
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8",
                                            "last-modified": "Wed, 21 Oct 2026 07:28:00 GMT"}, content=PAGE.encode())
    doc = fetch_url("https://shop.example/parts", client=client(handler))
    assert doc.source == "https://shop.example/parts"
    assert "back within three weeks" in doc.text and "newsletter" not in doc.text
    assert doc.metadata["kind"] == "url" and doc.metadata["type"] == "html"
    assert doc.metadata["title"] == "Spare parts | Brightwater Bikes"
    assert doc.metadata["modified"] == "2026-10-21T07:28:00Z"


def test_fetch_pdf_by_content_type():
    pdf = make_pdf(["Brake pads wear out after about 2,000 km."])
    c = client(lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf))
    doc = fetch_url("https://x.example/download?id=7", client=c)
    assert "Brake pads" in doc.text and doc.metadata["type"] == "pdf"


def test_fetch_follows_redirect_records_final_url():
    def handler(req):
        if req.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://x.example/new"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"moved here")
    doc = fetch_url("https://x.example/old", client=client(handler))
    assert doc.source == "https://x.example/old" and doc.metadata["final_url"] == "https://x.example/new"
    assert doc.text == "moved here"


def test_fetch_404_raises_with_status():
    c = client(lambda req: httpx.Response(404, content=b"nope"))
    with pytest.raises(FetchError, match="404"):
        fetch_url("https://x.example/missing", client=c)
    assert issubclass(FetchError, LoadError)


def test_fetch_network_error_is_fetch_error():
    def handler(req):
        raise httpx.ConnectError("name not resolved")
    with pytest.raises(FetchError, match="name not resolved"):
        fetch_url("https://nowhere.example", client=client(handler))


def test_fetch_unsupported_type():
    c = client(lambda req: httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"))
    with pytest.raises(FetchError, match="unsupported content type 'image/png'"):
        fetch_url("https://x.example/logo", client=c)


def test_fetch_too_large():
    c = client(lambda req: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5000))
    with pytest.raises(FetchError, match="larger than"):
        fetch_url("https://x.example/big", client=c, max_bytes=1000)


def test_fetch_suffix_fallback_for_octet_stream():
    c = client(lambda req: httpx.Response(200, headers={"content-type": "application/octet-stream"},
                                          content=b"# Readme\n\nhello"))
    doc = fetch_url("https://x.example/README.md", client=c)
    assert doc.metadata["type"] == "markdown" and doc.metadata["title"] == "Readme"
