"""OCR for scanned pages: rapidocr (PaddleOCR models in ONNX, bundled in the wheel).

A scanned PDF page is a picture of text with no text layer, so every text
extractor returns nothing for it. OCR reads the picture. It's slow (about
1.4 s a page on a laptop CPU) and can misread, so it only runs on pages that
have almost no text of their own.
"""

from __future__ import annotations

_engine = None


def ocr_image(image) -> str:
    """Text lines of a PIL image, top to bottom."""
    global _engine
    import numpy as np

    if _engine is None:
        from rapidocr import RapidOCR  # imported here: ~1 s to load, and most runs never OCR

        _engine = RapidOCR()
    result = _engine(np.array(image.convert("RGB")))
    return "\n".join(result.txts or ())
