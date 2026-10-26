import io
import json
from types import SimpleNamespace as NS

import pytest
from PIL import Image, ImageDraw

from raggy.config import Config
from raggy.generate.anthropic_llm import AnthropicLLM
from raggy.generate.fake import FakeLLM
from raggy.generate.llm import LLMError
from raggy.generate.openai_llm import OpenAILLM
from raggy.ingest.images import VISION_VERSION, ImageError, ImageReader, prepare_image
from raggy.pipeline import pipeline_key


def png(w=400, h=300, mode="RGB", text=None, fmt="PNG", **save) -> bytes:
    im = Image.new(mode, (w, h), "white" if mode in ("RGB", "RGBA", "CMYK") else 0)
    if text:
        ImageDraw.Draw(im).text((20, 20), text, fill="black")
    buf = io.BytesIO()
    im.save(buf, fmt, **save)
    return buf.getvalue()


def fake_ocr(lines):
    return lambda image: "\n".join(lines)


def test_prepare_image_normalises_modes_and_size():
    for data in (png(mode="P", transparency=0), png(mode="CMYK", fmt="JPEG"), png(mode="RGBA"),
                 png(mode="L", fmt="GIF")):
        out, w, h = prepare_image(data)
        im = Image.open(io.BytesIO(out))
        assert im.format == "PNG" and im.mode == "RGB" and (w, h) == (400, 300)
    out, w, h = prepare_image(png(6000, 3000))
    assert (w, h) == (1568, 784)  # long edge capped, aspect kept
    with pytest.raises(ImageError):
        prepare_image(b"<svg xmlns='http://www.w3.org/2000/svg'/>")


def test_tiny_image_is_decorative():
    assert prepare_image(png(64, 400)) is None  # short side under 100 px: an icon or a rule
    assert ImageReader("ocr", ocr=fake_ocr(["x"])).read(png(64, 64), "logo") is None


def test_ocr_mode_block():
    r = ImageReader("ocr", ocr=fake_ocr(["Jan 42", "Feb 57"]))
    assert r.read(png(), "chart.png") == "[Image: chart.png]\nText in image:\nJan 42\nFeb 57"
    blank = ImageReader("ocr", ocr=fake_ocr([]))
    assert blank.read(png(), "photo.png") == "[Image: photo.png]"  # the header alone: findable, nothing invented


def test_vision_mode_block_has_description_and_ocr():
    llm = FakeLLM("A bar chart of bikes serviced per month.")
    r = ImageReader("vision", llm=llm, ocr=fake_ocr(["Jan 42"]))
    block = r.read(png(), "chart.png")
    assert block == "[Image: chart.png]\nA bar chart of bikes serviced per month.\nText in image:\nJan 42"
    assert llm.image_counts == [1] and "chart.png" in llm.calls[0][1]  # the label gives the model context
    assert (r.calls, r.failures) == (1, 0)


def test_vision_failure_falls_back_to_ocr_and_counts(tmp_path):
    llm = FakeLLM(error=LLMError("ollama request failed"))
    r = ImageReader("vision", llm=llm, cache_path=tmp_path / "v.sqlite", ocr=fake_ocr(["Jan 42"]))
    assert r.read(png(), "c.png") == "[Image: c.png]\nText in image:\nJan 42"
    assert r.failures == 1
    empty = ImageReader("vision", llm=FakeLLM("   "), cache_path=tmp_path / "v.sqlite", ocr=fake_ocr([]))
    assert empty.read(png(), "c.png") == "[Image: c.png]" and empty.failures == 1  # an empty reply is a failure too
    ok = ImageReader("vision", llm=FakeLLM("described"), cache_path=tmp_path / "v.sqlite", ocr=fake_ocr([]))
    assert ok.read(png(), "c.png") == "[Image: c.png]\ndescribed" and ok.cached == 0  # failures weren't cached


def test_vision_cache_hit_skips_llm(tmp_path):
    cache = tmp_path / "vision-cache.sqlite"
    first = FakeLLM("described once")
    ImageReader("vision", llm=first, cache_path=cache, ocr=fake_ocr([])).read(png(text="a"), "a.png")
    again = FakeLLM("should not be called")
    r = ImageReader("vision", llm=again, cache_path=cache, ocr=fake_ocr([]))
    assert r.read(png(text="a"), "other-name.png") == "[Image: other-name.png]\ndescribed once"  # keyed by bytes
    assert again.calls == [] and r.cached == 1
    other_model = FakeLLM("new model's view")
    other_model.name = "fake:other"
    r2 = ImageReader("vision", llm=other_model, cache_path=cache, ocr=fake_ocr([]))
    assert r2.read(png(text="a"), "a.png").endswith("new model's view")  # the model is part of the key


class FakeOpenAI:
    def __init__(self):
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return NS(choices=[NS(message=NS(content="ok"))], usage=None)


def test_openai_adapter_sends_image_parts():
    llm = OpenAILLM("gpt-x")
    llm._client = FakeOpenAI()
    llm.complete("sys", "describe", images=[b"\x89PNG-bytes"])
    content = llm._client.requests[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    llm.complete("sys", "plain")
    assert llm._client.requests[1]["messages"][1]["content"] == "plain"  # no images: request unchanged


class FakeAnthropic:
    def __init__(self):
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="ok")],
                  usage=NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def test_anthropic_adapter_sends_image_blocks():
    llm = AnthropicLLM()
    llm._client = FakeAnthropic()
    llm.complete("sys", "describe", images=[b"\x89PNG-bytes"])
    content = llm._client.requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["type"] == "base64" and content[-1] == {"type": "text", "text": "describe"}


def test_images_settings_in_pipeline_key(tmp_path):
    base = dict(index_dir=tmp_path, embed_model="fake", llm_provider="openai", llm_model="gpt-x")
    keys = {pipeline_key(Config(**base, images=m), "e") for m in ("off", "ocr", "vision")}
    assert len(keys) == 3
    a = json.loads(pipeline_key(Config(**base, images="vision", vision_model="m1"), "e"))
    b = json.loads(pipeline_key(Config(**base, images="vision", vision_model="m2"), "e"))
    assert a != b and a["vision"]["version"] == VISION_VERSION
    off = json.loads(pipeline_key(Config(**base, images="ocr", vision_model="m1"), "e"))
    assert off["vision"] is None  # the vision model can't change chunks unless vision is on


def test_vision_model_follows_answer_model(monkeypatch, tmp_path):
    for k in ("RAGGY_VISION", "RAGGY_VISION_MODEL", "RAGGY_IMAGES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAGGY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("RAGGY_LLM", "openai")
    monkeypatch.setenv("RAGGY_LLM_MODEL", "gpt-x")
    c = Config.from_env()
    assert (c.images, c.vision_provider, c.vision_model) == ("off", "openai", "gpt-x")
    monkeypatch.setenv("RAGGY_VISION", "ollama")
    c = Config.from_env()
    assert (c.vision_provider, c.vision_model) == ("ollama", None)  # the adapter's default (qwen3.5:4b)
