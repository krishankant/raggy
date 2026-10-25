"""HTML: keep the article, drop the page around it.

A web page is mostly not its content: navigation, cookie banners, "related
posts", footers. Embedded as-is, that boilerplate lands in every chunk of
every page, and a query about returns matches the "Returns" link in the
menu of forty pages. ``main`` mode asks trafilatura for the main content as
Markdown (headings, lists and tables survive). ``full`` mode keeps every
visible word: it exists so the lesson can measure what extraction is worth.
"""

from __future__ import annotations

from html.parser import HTMLParser

from raggy.ingest import loaders

_SKIP = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
          "article", "header", "footer", "nav", "main", "aside", "table", "ul", "ol", "td", "th"}


class _AllText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP:
            self._skip += 1
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP and self._skip:
            self._skip -= 1
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")  # old pages; trafilatura sniffs better in main mode


def _full(data: bytes) -> tuple[str, str | None]:
    p = _AllText()
    p.feed(_decode(data))
    p.close()
    lines = (" ".join(line.split()) for line in "".join(p.parts).split("\n"))
    return "\n".join(line for line in lines if line), " ".join("".join(p.title).split()) or None


def load(data: bytes, options) -> tuple[str, str | None]:
    if options.html_extract == "full":
        return _full(data)
    import trafilatura  # imported here: it pulls in lxml, which `raggy ask` never needs

    # extract(), not bare_extraction(): in trafilatura 2.x only extract() renders Markdown.
    body = trafilatura.extract(data, output_format="markdown", include_tables=True,
                               include_comments=False, include_images=False)
    if not (body or "").strip():
        raise loaders.LoadError("no text found (the page may need JavaScript to render)")
    _, title = _full(data)  # the page's own <title>
    return body, title
