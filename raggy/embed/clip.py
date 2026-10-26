"""CLIP: images and text in one vector space, for finding pictures (week 10, optional).

The text index finds an image through the text written about it (OCR or a
description). CLIP finds it by what it looks like: a photo of a cargo bike
lands near the words "cargo bike" and near other photos of cargo bikes,
with no description at all. It was trained on photos with captions, so it's
strongest on photos and weak on charts, diagrams and text in images.
"""

from __future__ import annotations

import io

import numpy as np

VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"  # 0.34 GB, MIT, 512-d
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"  # its paired text tower


class ClipEmbedder:
    dim = 512

    def __init__(self, vision_model: str = VISION_MODEL, text_model: str = TEXT_MODEL):
        self.name = f"clip:{vision_model}"
        self._vision_model, self._text_model = vision_model, text_model
        self._vision = self._text = None  # loaded on first use

    @staticmethod
    def _norm(vectors) -> np.ndarray:
        v = np.asarray(list(vectors), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def embed_images(self, images: list[bytes]) -> np.ndarray:
        from fastembed import ImageEmbedding
        from PIL import Image

        if self._vision is None:
            self._vision = ImageEmbedding(self._vision_model)
        return self._norm(self._vision.embed([Image.open(io.BytesIO(b)).convert("RGB") for b in images]))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        from fastembed import TextEmbedding

        if self._text is None:
            self._text = TextEmbedding(self._text_model)
        return self._norm(self._text.embed(texts))
