from pathlib import Path


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    """A tiny, valid PDF with one line of Helvetica text per page, written by hand.

    Hand-written so tests need no PDF-writing library; pypdf reads it like any other.
    """
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font = 3 + 2 * n
    for i, text in enumerate(pages):
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    info = None
    if title:
        objs.append(f"<< /Title ({title}) >>".encode())
        info = len(objs)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    trailer = f"<< /Size {len(objs) + 1} /Root 1 0 R" + (f" /Info {info} 0 R" if info else "") + " >>"
    out += f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def write(p: Path, text: str | bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    return p


PAGE = """<html><head><title>Spare parts | Brightwater Bikes</title></head><body>
<nav><ul><li><a href="/">Home</a></li><li><a href="/bikes">Bikes</a></li><li><a href="/returns">Returns and refunds</a></li></ul></nav>
<div class="cookie">We use cookies to improve your experience. Accept all cookies or manage preferences.</div>
<main><article><h1>Spare parts catalogue</h1>
<p>Order parts by their code. Codes start with BW for Brightwater-made parts and XP for parts we resell from other makers.</p>
<table><thead><tr><th>Code</th><th>Part</th></tr></thead><tbody>
<tr><td>BW-CH-1142</td><td>8-speed chain</td></tr></tbody></table>
<p>Out-of-stock parts are usually back within three weeks, and we email you when they arrive at the warehouse.</p>
</article></main>
<script>var tracking = "do not index me";</script>
<footer><p>Sign up to our newsletter for ten percent off your next order.</p></footer></body></html>"""
