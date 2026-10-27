import io
from fractions import Fraction

import numpy as np
import pytest

from raggy.config import Config
from raggy.generate.prompt import location
from raggy.ingest.loaders import LoadError, LoadOptions, load_bytes, load_file
from raggy.ingest.media import Segment, Transcriber, fmt_time, parse_subtitles, windows
from raggy.pipeline import pipeline_key

TALK = [Segment(0.0, 14.0, "Welcome to the Brightwater podcast.", "Emily"),
        Segment(14.0, 40.0, "Today we're talking about the Ebb E-Cargo.", "Emily"),
        Segment(40.0, 75.0, "The battery is guaranteed for two years.", "Paul"),
        Segment(75.0, 130.0, "And the range is about seventy kilometres with a full box.", "Paul")]


def test_fmt_time():
    assert [fmt_time(t) for t in (0, 83.4, 600, 3725)] == ["00:00", "01:23", "10:00", "1:02:05"]


def test_windows_split_by_time_and_speaker():
    ws = windows(TALK, 60)
    assert [[s.text[:7] for s in w] for w in ws] == [["Welcome", "Today w"], ["The bat"], ["And the"]]
    one_long = [Segment(0, 200, "A very long monologue.")]
    assert windows(one_long, 60) == [one_long]  # never split mid-segment: the window just runs long


def test_parse_srt_and_vtt():
    srt = "1\n00:00:01,000 --> 00:00:04,000\n<i>Emily:</i> Hello and welcome.\n\n2\n00:00:04,000 --> 00:00:09,500\nPaul: Thanks, Emily.\n"
    segs = parse_subtitles(srt)
    assert [(s.start, s.end, s.speaker, s.text) for s in segs] == [
        (1.0, 4.0, "Emily", "Hello and welcome."), (4.0, 9.5, "Paul", "Thanks, Emily.")]
    vtt = ("WEBVTT\n\n00:00.000 --> 00:03.000\n<v Emily>Our launch day is 21 April.</v>\n\n"
           "00:03.000 --> 00:05.000\n<v Emily>Our launch day is 21 April.</v>\n\n"  # a rolling-caption repeat
           "01:02:03.500 --> 01:02:06.000\n<v Paul>See you then.\n")
    segs = parse_subtitles(vtt)
    assert [(s.speaker, s.text) for s in segs] == [("Emily", "Our launch day is 21 April."), ("Paul", "See you then.")]
    assert segs[-1].start == 3723.5


def test_subtitle_file_loads_with_time_locations(tmp_path):
    p = tmp_path / "episode.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:04,000\nEmily: The workshop opens at nine.\n\n"
                 "2\n00:01:10,000 --> 00:01:14,000\nPaul: And closes at six.\n")
    doc = load_file(p)  # no model needed: subtitles always load
    assert doc.metadata["type"] == "subtitles"
    assert [label for label, _ in doc.metadata["pages"]] == ["@ 00:01", "@ 01:10"]
    assert "## 00:01–00:04 · Emily" in doc.text and "Emily: The workshop opens at nine." in doc.text
    assert location({"page": "@ 00:01", "page_end": "@ 01:10"}) == "@ 00:01–01:10"


def fake_transcribe(segments):
    calls = []

    def run(data: bytes, name: str):
        calls.append(name)
        return list(segments)
    run.calls = calls
    return run


def test_audio_file_needs_media_on(tmp_path):
    p = tmp_path / "talk.mp3"
    p.write_bytes(b"ID3fake")
    with pytest.raises(LoadError, match="RAGGY_MEDIA=on"):
        load_file(p)


def test_audio_file_transcribed_into_time_windows(tmp_path):
    p = tmp_path / "talk.mp3"
    p.write_bytes(b"ID3fake")
    t = fake_transcribe(TALK)
    doc = load_file(p, LoadOptions(media="on", transcribe=t))
    assert t.calls == ["talk.mp3"] and doc.metadata["type"] == "audio"
    assert [label for label, _ in doc.metadata["pages"]] == ["@ 00:00", "@ 00:40", "@ 01:15"]
    assert "Paul: The battery is guaranteed for two years." in doc.text


def test_silent_file_is_an_error(tmp_path):
    p = tmp_path / "silence.wav"
    p.write_bytes(b"RIFFfake")
    with pytest.raises(LoadError, match="no speech found"):
        load_file(p, LoadOptions(media="on", transcribe=fake_transcribe([])))


def test_transcriber_caches_by_bytes_and_model(tmp_path):
    runs = []

    class FakeWhisper:
        def transcribe(self, audio, **kw):
            runs.append(kw)
            return iter([type("S", (), {"start": 0.0, "end": 2.0, "text": " Hello. "})()]), None

    cache = tmp_path / "transcript-cache.sqlite"
    t = Transcriber("base", cache_path=cache, model=FakeWhisper())
    assert t.transcribe(b"audio-1", "a.mp3") == [Segment(0.0, 2.0, "Hello.")]
    again = Transcriber("base", cache_path=cache, model=FakeWhisper())
    assert again.transcribe(b"audio-1", "renamed.mp3") == [Segment(0.0, 2.0, "Hello.")] and len(runs) == 1
    other = Transcriber("small", cache_path=cache, model=FakeWhisper())
    other.transcribe(b"audio-1", "a.mp3")
    assert len(runs) == 2  # another model is another transcript


def tiny_video(seconds=4, fps=2, colours=((200, 30, 30), (200, 30, 30), (30, 60, 220), (30, 60, 220))) -> bytes:
    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
        s.time_base = Fraction(1, fps)
        for i in range(seconds * fps):
            rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            rgb[:] = colours[min(i // fps, len(colours) - 1)]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for packet in s.encode(frame):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    return buf.getvalue()


def test_video_keyframes_placed_in_their_windows():
    labels = []

    def reader(data: bytes, label: str):
        labels.append(label)
        return f"[Image: {label}]\nA coloured frame."

    segs = [Segment(0.0, 1.5, "Here is the red one.", "Emily"), Segment(1.5, 4.0, "And now the blue one.", "Paul")]
    opts = LoadOptions(media="on", images="ocr", read_image=reader, transcribe=fake_transcribe(segs),
                       keyframe_seconds=1)
    doc = load_bytes(tiny_video(), "/x/clip.mp4", "video", opts)
    assert labels == ["clip.mp4 @ 00:00, frame", "clip.mp4 @ 00:02, frame"]  # the repeated red/blue frames skipped
    t = doc.text
    assert t.index("Here is the red one") < t.index("[Image: clip.mp4 @ 00:00, frame]") < t.index("## 00:01")
    assert t.index("[Image: clip.mp4 @ 00:02, frame]") > t.index("And now the blue one")


def test_media_settings_in_pipeline_key(tmp_path):
    base = dict(index_dir=tmp_path, embed_model="fake", llm_provider="fake", llm_model=None)
    keys = {pipeline_key(Config(**base, **kw), "e") for kw in
            ({}, {"media": "on"}, {"media": "on", "whisper_model": "small"}, {"media": "on", "media_window": 30})}
    assert len(keys) == 4


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: an English caption track, or none."""

    captions = True

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        info = {"title": "Ebb launch livestream", "duration": 120, "webpage_url": url}
        if self.captions:
            info["subtitles"] = {"en": [{"ext": "vtt", "url": "https://example.test/en.vtt"}]}
        return info


def test_youtube_url_uses_published_captions(monkeypatch):
    import sys
    import types

    from raggy.ingest import fetch

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setattr(fetch, "_get_text", lambda url: "WEBVTT\n\n00:05.000 --> 00:09.000\n<v Emily>Launch day is 21 April.\n")
    doc = fetch.fetch_url("https://www.youtube.com/watch?v=abc123")
    assert doc.metadata["title"] == "Ebb launch livestream" and doc.metadata["type"] == "subtitles"
    assert "Emily: Launch day is 21 April." in doc.text and doc.metadata["pages"][0][0] == "@ 00:05"


def test_media_url_without_yt_dlp_is_a_load_error(monkeypatch):
    import sys

    from raggy.ingest import fetch
    from raggy.ingest.loaders import LoadError

    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # import fails
    with pytest.raises(LoadError, match="uv sync --extra media"):
        fetch.fetch_url("https://youtu.be/abc123")


def test_media_suite_captions_resolve_and_media_off_loses_the_rest():
    from pathlib import Path

    from raggy.evals.dataset import resolve_gold
    from raggy.evals.suites import SUITES, media_suite

    s = media_suite(Path(__file__).resolve().parents[2] / "evals" / "media")
    qs = s.questions()
    assert "media" in SUITES and len(qs) == 18 and s.strict_gold is False
    missing: list[str] = []
    resolve_gold(qs, s.documents(LoadOptions()), strict=False, missing=missing)  # media off: captions only
    assert sorted(missing) == sorted(q.id for q in qs if q.answerable and not q.id.startswith("c"))
