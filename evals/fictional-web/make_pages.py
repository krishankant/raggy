"""Build the fictional-web suite: the fictional docs as pages of a web site.

Each Markdown doc in ../fictional/docs becomes an HTML page inside the same
site template a real shop would use: a cookie banner, a header with a big
navigation menu and a search box, a breadcrumb, a "popular help articles"
sidebar, and a footer with a newsletter form and link columns. Two extra
pages are copies, as real sites have:

- help/shipping-and-delivery.html: the shipping article again, at a second
  URL, in the site's "help centre" layout.
- archive/returns-and-refunds-2025.html: last year's returns page, archived,
  identical except that the replacement box cost 20 credits then.

The questions are the fictional suite's, so every gold quote still points
at real text. Run from the repo root:  uv run python evals/fictional-web/make_pages.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE.parent / "fictional" / "docs"
OUT = HERE / "pages"

NAV = {
    "Bikes": ["Heron Commuter", "Heron Tourer", "Kestrel Gravel", "Wren Kids 20", "Ebb E-Cargo", "Compare bikes"],
    "Shop": ["Accessories", "Spare parts", "Helmets", "Lights", "Locks", "Gift cards"],
    "Help": ["Returns and refunds", "Warranty", "Shipping and delivery", "Service centres", "Bike fitting",
             "Contact us"],
    "About": ["Our story", "Careers", "Cycle-to-work scheme", "Press", "Sustainability"],
}
POPULAR = ["Returns and refunds explained", "Looking after your e-bike battery", "Find your nearest service centre",
           "Our frame size guide", "Delivery times and costs", "How the cycle-to-work scheme works"]


def _inline(text: str) -> str:
    return html.escape(text, quote=False)


def md_to_html(md: str) -> tuple[str, str]:
    """The small subset of Markdown the fictional docs use: headings, paragraphs, lists, tables."""
    title, out, para, items, rows = "", [], [], [], []

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if rows:
            head, *body = rows
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            rows.clear()

    for line in md.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif m := re.match(r"^(#{1,6})\s+(.*)$", s):
            flush()
            level = len(m.group(1))
            title = title or m.group(2)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif s.startswith("- "):
            if para or rows:
                flush()
            items.append(s[2:])
        elif s.startswith("|"):
            if para or items:
                flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        else:
            if items or rows:
                flush()
            para.append(s)
    flush()
    return title, "\n".join(out)


def _nav() -> str:
    return "\n".join(
        f'<li class="menu"><a href="#">{_inline(group)}</a><ul>'
        + "".join(f'<li><a href="#">{_inline(x)}</a></li>' for x in links) + "</ul></li>"
        for group, links in NAV.items())


def page(title: str, body: str, layout: str = "shop", notice: str = "") -> str:
    sidebar = "" if layout == "help" else (
        '<aside class="related"><h2>Popular help articles</h2><ul>'
        + "".join(f'<li><a href="#">{_inline(x)}</a></li>' for x in POPULAR)
        + "</ul><p>Still need help? Our customer team answers messages Monday to Friday. "
          "Send us a message from your account page and we will get back to you.</p></aside>")
    crumb = "Help centre" if layout == "help" else "Help"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_inline(title)} | Brightwater Bikes</title>
<link rel="stylesheet" href="/static/site.css">
<script>window.dataLayer = window.dataLayer || []; dataLayer.push({{"page": "{layout}"}});</script>
</head>
<body class="{layout}">
<div class="cookie-banner"><p>We use cookies to give you the best experience on our website and to show you
relevant offers. By continuing to browse you agree to our use of cookies.</p>
<button>Accept all cookies</button> <button>Manage preferences</button></div>
<header>
<a class="logo" href="/">Brightwater Bikes</a>
<nav class="main-menu"><ul>
{_nav()}
</ul></nav>
<form class="search"><input type="search" placeholder="Search bikes, parts and help articles"><button>Search</button></form>
<div class="promo">Join the Brightwater Riders Club for early access to new models and members-only events.</div>
</header>
<main>
<nav class="breadcrumb"><a href="/">Home</a> › <a href="#">{crumb}</a> › {_inline(title)}</nav>
{notice}
<article>
{body}
</article>
{sidebar}
</main>
<footer>
<div class="newsletter"><h2>Ride with us</h2><p>Sign up to our newsletter for route ideas, maintenance tips and
news about new bikes.</p><form><input type="email" placeholder="Your email address"><button>Sign up</button></form></div>
<div class="columns">
<ul><li><a href="#">Bikes</a></li><li><a href="#">Accessories</a></li><li><a href="#">Spare parts</a></li></ul>
<ul><li><a href="#">Returns and refunds</a></li><li><a href="#">Warranty</a></li><li><a href="#">Shipping and delivery</a></li><li><a href="#">Service centres</a></li></ul>
<ul><li><a href="#">Our story</a></li><li><a href="#">Careers</a></li><li><a href="#">Privacy policy</a></li><li><a href="#">Cookie settings</a></li></ul>
</div>
<p class="legal">© 2026 Brightwater Bikes. All prices are in credits and include tax.</p>
</footer>
</body>
</html>
"""


def render_all(docs: Path = DOCS) -> dict[str, str]:
    """Every page, keyed by its path under pages/."""
    pages = {}
    for f in sorted(docs.glob("*.md")):
        md = f.read_text(encoding="utf-8")
        title, body = md_to_html(md)
        pages[f.stem + ".html"] = page(title, body)
        if f.stem == "shipping":
            pages["help/shipping-and-delivery.html"] = page(title, body, layout="help")
        if f.stem == "returns-and-refunds":
            old = md.replace("replacement box for 25 credits", "replacement box for 20 credits")
            assert old != md, "the archived page must differ from the current one"
            t, b = md_to_html(old)
            pages["archive/returns-and-refunds-2025.html"] = page(
                t + " (2025)", b,
                notice='<div class="archived">This page is archived and may be out of date.</div>')
    return pages


def main() -> None:
    for rel, text in render_all().items():
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(f"wrote {len(render_all())} pages to {OUT}")


if __name__ == "__main__":
    main()
