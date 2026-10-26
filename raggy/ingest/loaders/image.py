"""A picture file (PNG, JPEG, WebP, GIF, BMP, TIFF) as one text block (week 10).

The text comes from the image reader the options carry: OCR, or a vision
model's description plus OCR. With images off, a picture has no text at all,
so loading it is an error that names the switch, never a silent skip.
"""

from __future__ import annotations

from raggy.ingest import loaders
from raggy.ingest.images import ImageError, read_image


def load(data: bytes, options) -> tuple[str, str | None]:
    if options.images == "off":
        raise loaders.LoadError("an image: set RAGGY_IMAGES=ocr or vision to read images")
    name = options.name or "image"
    try:
        block = read_image(options, data, name)
    except ImageError as e:
        raise loaders.LoadError(str(e)) from e
    if block is None:
        raise loaders.LoadError("image too small to carry information (decorative); skipped")
    if options.on_image is not None:
        options.on_image(data, name, None)
    return block, None
