import io

import numpy as np
import pytest
from PIL import Image

from raggy.config import Config, ConfigError
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.pipeline import Raggy

COLOURS = {"red": (220, 30, 30), "green": (30, 200, 60), "blue": (30, 60, 220)}


class FakeClip:
    """Images and texts in one 3-d space: an image is its mean colour, a text is the colour words it names."""

    name = "fake-clip"
    dim = 3

    def embed_images(self, images: list[bytes]) -> np.ndarray:
        out = []
        for data in images:
            v = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = sum((np.array(rgb, dtype=np.float32) for w, rgb in COLOURS.items() if w in t.lower()),
                    np.ones(3, dtype=np.float32))
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def solid(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (300, 200), COLOURS[colour]).save(path)
    return path


def make(tmp_path, monkeypatch, image_search=True):
    monkeypatch.setattr("raggy.ingest.ocr.ocr_image", lambda image: "")
    c = Config(index_dir=tmp_path / "i", embed_model="fake", llm_provider="fake", llm_model=None,
               images="ocr", image_search=image_search)
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM(), clip=FakeClip())


def test_search_images_by_text(tmp_path, monkeypatch):
    for colour in COLOURS:
        solid(tmp_path / "docs" / f"{colour}.png", colour)
    r = make(tmp_path, monkeypatch)
    r.ingest([str(tmp_path / "docs")])
    hits = r.search_images(text="the blue one", k=2)
    assert [h.label for h in hits][0] == "blue.png" and hits[0].source.endswith("blue.png")
    assert hits[0].score > hits[1].score


def test_search_images_by_image(tmp_path, monkeypatch):
    for colour in COLOURS:
        solid(tmp_path / "docs" / f"{colour}.png", colour)
    r = make(tmp_path, monkeypatch)
    r.ingest([str(tmp_path / "docs")])
    query = solid(tmp_path / "q" / "query.png", "green").read_bytes()
    assert r.search_images(image=query, k=1)[0].label == "green.png"


def test_remove_source_removes_images(tmp_path, monkeypatch):
    for colour in COLOURS:
        solid(tmp_path / "docs" / f"{colour}.png", colour)
    r = make(tmp_path, monkeypatch)
    r.ingest([str(tmp_path / "docs")])
    r.remove(str(tmp_path / "docs" / "blue.png"))
    assert "blue.png" not in [h.label for h in r.search_images(text="blue", k=5)]
    (tmp_path / "docs" / "red.png").unlink()
    r.ingest([str(tmp_path / "docs")])  # a file gone from the folder leaves the image index too
    # (blue.png is back: `raggy remove` drops a file from the index, not from disk, and the folder still has it)
    assert [h.label for h in r.search_images(text="blue", k=5)] == ["blue.png", "green.png"]


def test_image_search_off_writes_nothing(tmp_path, monkeypatch):
    solid(tmp_path / "docs" / "red.png", "red")
    r = make(tmp_path, monkeypatch, image_search=False)
    r.ingest([str(tmp_path / "docs")])
    with pytest.raises(ConfigError, match="RAGGY_IMAGE_SEARCH"):
        r.search_images(text="red")


def test_image_search_needs_images_on(tmp_path):
    with pytest.raises(ConfigError, match="RAGGY_IMAGES"):
        Config(index_dir=tmp_path, embed_model="fake", llm_provider="fake", llm_model=None, image_search=True)


def test_images_command_explains_when_off(tmp_path, monkeypatch, capsys):
    from raggy.cli import main

    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path / "i"))
    monkeypatch.delenv("RAGGY_IMAGE_SEARCH", raising=False)
    assert main(["images", "a blue cargo bike"]) == 1
    assert "RAGGY_IMAGE_SEARCH=on" in capsys.readouterr().err
