"""Images as searchable text: OCR, and optionally a vision model's description.

An embedding model indexes text, so a picture is only findable through text
about it. Two readers write that text:

- OCR (week 9's engine) transcribes the words in the image exactly, but loses
  how they relate: which number sits on which bar, which arrow points where.
- A vision model describes the picture: its kind, title, labels and numbers,
  and what's drawn. It catches relations and things with no words (a colour,
  a shape), but it's a model's paraphrase and can misread, so the exact OCR
  text is kept beside it.

Every image becomes one block, always starting with the same header:

    [Image: ebb-spec-sheet.pdf p. 2, figure 1]
    <description>
    Text in image:
    <OCR lines>

Results are cached by image bytes + mode + model + prompt version, so an
image is read once however many times it's re-ingested.
"""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path
from typing import Callable

from raggy.generate.llm import LLM, LLMError

VISION_VERSION = "vis-v1"
VISION_CACHE = "vision-cache.sqlite"
MIN_SIDE = 100  # smaller than this on its short side: an icon, a bullet, a rule — skipped
MAX_EDGE = 1568  # long edge sent to models; larger costs time and tokens without more detail
VISION_SYSTEM = """You describe images from a company's documents so they can be found by search.
Write plain text, under 150 words:
- what kind of image it is (chart, diagram, photo, screenshot, label, map, drawing)
- its title, if it has one
- every label and number in it, copied exactly, keeping each value with what it belongs to
- what is shown: objects, colours, arrows and connections, positions
Describe only what is visible. Do not guess what isn't shown."""


class ImageError(Exception):
    """Bytes that aren't a decodable raster image."""


def prepare_image(data: bytes) -> tuple[bytes, int, int] | None:
    """Decode, convert to RGB, cap the long edge; PNG bytes and the new size.
    None for a decorative image (too small to carry information)."""
    from PIL import Image, UnidentifiedImageError

    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)  # an animated GIF: its first frame
        im.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise ImageError(f"not a readable image ({type(e).__name__})") from e
    if min(im.size) < MIN_SIDE:
        return None
    if im.mode in ("RGBA", "LA", "P", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")  # transparent pixels become white, not black
        im = Image.new("RGB", rgba.size, "white")
        im.paste(rgba, mask=rgba.getchannel("A"))
    elif im.mode != "RGB":
        im = im.convert("RGB")  # CMYK, greyscale, 16-bit…
    if max(im.size) > MAX_EDGE:
        scale = MAX_EDGE / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue(), im.width, im.height


class VisionCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS images (key TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self.conn.commit()
        import threading

        self._lock = threading.Lock()  # ingest loads files in parallel (week 12)

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT body FROM images WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO images (key, body) VALUES (?, ?)", (key, body))
            self.conn.commit()


def _default_ocr(image) -> str:
    from raggy.ingest.ocr import ocr_image

    return ocr_image(image)


class ImageReader:
    """mode "ocr": the text in the image; "vision": a description plus that text."""

    def __init__(self, mode: str, llm: LLM | None = None, cache_path: str | Path | None = None,
                 ocr: Callable = _default_ocr):
        if mode not in ("ocr", "vision"):
            raise ValueError(f"image mode must be ocr or vision, not {mode!r}")
        if mode == "vision" and llm is None:
            raise ValueError("vision mode needs a model")
        self.mode, self.llm, self._ocr = mode, llm, ocr
        self.cache = VisionCache(cache_path) if cache_path else None
        self.calls = self.cached = self.failures = 0  # model calls, cache hits, failed descriptions

    def _key(self, data: bytes) -> str:
        model = self.llm.name if self.mode == "vision" else "-"
        return hashlib.sha256(hashlib.sha256(data).digest() + f"|{self.mode}|{model}|{VISION_VERSION}".encode()).hexdigest()

    def read(self, data: bytes, label: str) -> str | None:
        """The image's text block, or None if it's decorative. Raises ImageError on bad bytes."""
        prepared = prepare_image(data)
        if prepared is None:
            return None
        header = f"[Image: {label}]"
        key = self._key(data)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.cached += 1
            return f"{header}\n{hit}" if hit else header
        from PIL import Image

        png = prepared[0]
        text = self._ocr(Image.open(io.BytesIO(png))).strip()
        parts = []
        ok = True
        if self.mode == "vision":
            self.calls += 1
            try:
                description = " ".join(self.llm.complete(VISION_SYSTEM, f"Image: {label}", images=[png]).split())
            except LLMError:
                description = ""
            if description:
                parts.append(description)
            else:  # fall back to the OCR text, count it, and don't cache it: the next ingest retries
                self.failures += 1
                ok = False
        if text:
            parts.append("Text in image:\n" + text)
        body = "\n".join(parts)
        if ok and self.cache is not None:
            self.cache.put(key, body)
        return f"{header}\n{body}" if body else header


def read_image(options, data: bytes, label: str) -> str | None:
    """What a loader calls for each image: None if images are off or the image is decorative."""
    if getattr(options, "images", "off") == "off":
        return None
    reader = options.read_image
    if reader is None:  # loaded outside a Raggy (a script, a test): OCR without a cache
        global _plain_ocr
        if _plain_ocr is None:
            _plain_ocr = ImageReader("ocr")
        reader = _plain_ocr.read
    return reader(data, label)


_plain_ocr: ImageReader | None = None


class FigureReader:
    """Reads the pictures inside one document: each distinct image once (a logo on every
    slide is read at its first appearance), and a bad one is skipped with a warning
    instead of failing the whole document."""

    def __init__(self, options):
        self.options = options
        self.name = getattr(options, "name", "") or "document"
        self.on = getattr(options, "images", "off") != "off"
        self._seen: set[str] = set()

    def read(self, data: bytes, label: str, page: str | None = None) -> str | None:
        if not self.on:
            return None
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._seen:
            return None
        self._seen.add(digest)
        try:
            block = read_image(self.options, data, label)
        except ImageError as e:
            import warnings

            warnings.warn(f"{self.name}: {label} skipped ({e})", UserWarning, stacklevel=2)
            return None
        if block is not None and self.options.on_image is not None:
            self.options.on_image(data, label, page)
        return block
