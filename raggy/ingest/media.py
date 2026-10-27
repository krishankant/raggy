"""Audio and video as time-labelled text (week 11).

Speech is text with timestamps. A transcriber (faster-whisper, local) turns
the audio into segments, `Segment(start, end, text, speaker)`. The segments
are grouped into windows: at most RAGGY_MEDIA_WINDOW seconds, and a new
window at every speaker change. Each window becomes one labelled part:

    ## 01:23–02:10 · Paul

    Paul: The battery is guaranteed for two years. ...

labelled "@ 01:23", so week 9's location machinery stamps every chunk with
its time range and citations read `talk.mp3 @ 01:23–02:10`.

Video is its audio plus keyframes: one frame every RAGGY_KEYFRAME_SECONDS,
near-identical frames skipped, each read by week 10's image reader and
placed in the window that covers its time.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

TRANSCRIBE_VERSION = "tr-v1"
TRANSCRIPT_CACHE = "transcript-cache.sqlite"
MAX_FRAMES = 120  # per file: the frame step grows for long videos so ingest stays bounded


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float
    text: str
    speaker: str | None = None


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --- grouping ---------------------------------------------------------------

def windows(segments: list[Segment], seconds: float) -> list[list[Segment]]:
    """Consecutive segments, at most ``seconds`` long per group, a new group per speaker.
    A single segment longer than the window is never split: its window runs long."""
    out: list[list[Segment]] = []
    for seg in segments:
        cur = out[-1] if out else None
        if cur and seg.speaker == cur[0].speaker and seg.end - cur[0].start <= seconds:
            cur.append(seg)
        else:
            out.append([seg])
    return out


def render(groups: list[list[Segment]], frames: dict[int, list[str]] | None = None) -> list[tuple[str, str]]:
    """One labelled part per window: a heading with its time range, its lines, then its frames."""
    parts = []
    for i, w in enumerate(groups):
        speakers = {s.speaker for s in w}
        who = f" · {w[0].speaker}" if len(speakers) == 1 and w[0].speaker else ""
        head = f"## {fmt_time(w[0].start)}–{fmt_time(w[-1].end)}{who}"
        if w[0].speaker:
            body = "\n".join(f"{s.speaker}: {s.text}" for s in w)
        else:
            body = " ".join(s.text for s in w)
        blocks = (frames or {}).get(i, [])
        parts.append((f"@ {fmt_time(w[0].start)}", "\n\n".join([head, body, *blocks])))
    return parts


# --- subtitles --------------------------------------------------------------

_CUE = re.compile(r"((?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})")
_VOICE = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>")
_TAG = re.compile(r"<[^>]+>")
_PREFIX = re.compile(r"^([A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*)?):\s+(.+)$")


def _seconds(stamp: str) -> float:
    parts = stamp.replace(",", ".").split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def _speaker(line: str) -> tuple[str | None, str]:
    m = _VOICE.search(line)
    if m:
        return m.group(1).strip(), html.unescape(_TAG.sub("", line)).strip()
    text = html.unescape(_TAG.sub("", line)).strip()
    m = _PREFIX.match(text)
    return (m.group(1), m.group(2).strip()) if m else (None, text)


def parse_subtitles(text: str) -> list[Segment]:
    """SRT or WebVTT cues as segments: tags stripped, voice tags and "Name:" prefixes as speakers,
    and a cue that repeats the previous one (rolling captions) dropped."""
    segs: list[Segment] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _CUE.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = _seconds(m.group(1)), _seconds(m.group(2))
        i += 1
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        speaker, words = _speaker(" ".join(body))
        if not words:
            continue
        if segs and segs[-1].text == words and segs[-1].speaker == speaker:
            continue
        segs.append(Segment(start, end, words, speaker))
    return segs


# --- transcription ----------------------------------------------------------

def _prefixed(segments: list[Segment]) -> list[Segment]:
    out = []
    for s in segments:
        speaker, text = _speaker(s.text) if s.speaker is None else (s.speaker, s.text)
        out.append(Segment(s.start, s.end, text, speaker))
    return out


class Transcriber:
    """faster-whisper on the CPU (int8), cached by audio bytes + model + version."""

    def __init__(self, model_name: str = "base", cache_path: str | Path | None = None, model=None):
        self.model_name = model_name
        self._model = model  # built on first use: the first call downloads the model once
        self.calls = self.cached = 0
        self._db = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS transcripts (key TEXT PRIMARY KEY, segments TEXT NOT NULL)")
            self._db.commit()

    def _key(self, data: bytes) -> str:
        return hashlib.sha256(hashlib.sha256(data).digest() + f"|{self.model_name}|{TRANSCRIBE_VERSION}".encode()).hexdigest()

    def transcribe(self, data: bytes, name: str = "") -> list[Segment]:
        key = self._key(data)
        if self._db is not None:
            row = self._db.execute("SELECT segments FROM transcripts WHERE key = ?", (key,)).fetchone()
            if row:
                self.cached += 1
                return [Segment(**s) for s in json.loads(row[0])]
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self.calls += 1
        try:
            raw, _ = self._model.transcribe(io.BytesIO(data), vad_filter=True, beam_size=5)
            segs = [Segment(round(float(s.start), 2), round(float(s.end), 2), s.text.strip()) for s in raw]
        except Exception as e:  # PyAV raises its own error types for unreadable or audio-less files
            from raggy.ingest import loaders

            raise loaders.LoadError(f"can't read the audio ({type(e).__name__}: {e})") from e
        segs = [s for s in segs if s.text]
        if self._db is not None:
            self._db.execute("INSERT OR REPLACE INTO transcripts (key, segments) VALUES (?, ?)",
                             (key, json.dumps([asdict(s) for s in segs])))
            self._db.commit()
        return segs


_default: dict[str, Transcriber] = {}


def _transcribe(options, data: bytes) -> list[Segment]:
    fn = options.transcribe
    if fn is None:  # loaded outside a Raggy (a script, a test): uncached
        fn = _default.setdefault(options.whisper_model, Transcriber(options.whisper_model)).transcribe
    return _prefixed(fn(data, options.name))


# --- keyframes --------------------------------------------------------------

def keyframes(data: bytes, step: float) -> list[tuple[float, bytes]]:
    """(time, PNG) every ``step`` seconds (wider for long videos), skipping near-repeats."""
    import av
    from PIL import Image, ImageChops, ImageStat

    out, last = [], None
    with av.open(io.BytesIO(data)) as c:
        if not c.streams.video:
            return []
        stream = c.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else (c.duration or 0) / 1e6
        step = max(step, duration / MAX_FRAMES) if duration else step
        t = 0.0
        while t <= max(duration, 0.0):
            c.seek(int(t / stream.time_base), stream=stream, backward=True, any_frame=False)
            frame = None
            for f in c.decode(stream):
                if f.time is None or f.time >= t - 1e-6:
                    frame = f
                    break
            if frame is None:
                break
            im = frame.to_image().convert("RGB")
            thumb = im.resize((16, 16)).convert("L")
            if last is None or ImageStat.Stat(ImageChops.difference(thumb, last)).mean[0] > 6:
                buf = io.BytesIO()
                im.save(buf, "PNG")
                out.append((t, buf.getvalue()))
                last = thumb
            t += step
    return out


# --- the loaders --------------------------------------------------------------

def _need_media(options) -> None:
    from raggy.ingest import loaders

    if getattr(options, "media", "off") != "on":
        raise loaders.LoadError("audio or video: set RAGGY_MEDIA=on to transcribe it")


def load_audio(data: bytes, options):
    from raggy.ingest import loaders

    _need_media(options)
    segs = _transcribe(options, data)
    if not segs:
        raise loaders.LoadError("no speech found")
    return render(windows(segs, options.media_window)), None


def load_video(data: bytes, options):
    from raggy.ingest import loaders
    from raggy.ingest.images import FigureReader

    _need_media(options)
    try:
        segs = _transcribe(options, data)
    except loaders.LoadError:
        segs = []  # a silent video can still carry its frames
    groups = windows(segs, options.media_window)
    frames: dict[int, list[str]] = {}
    figures = FigureReader(options)
    if figures.on:
        for t, png in keyframes(data, options.keyframe_seconds):
            label = f"{figures.name} @ {fmt_time(t)}, frame"
            block = figures.read(png, label, f"@ {fmt_time(t)}")
            if block is None:
                continue
            if not groups:
                groups.append([Segment(t, t, "")])
            i = max((k for k, w in enumerate(groups) if w[0].start <= t), default=0)
            frames.setdefault(i, []).append(block)
    if not groups or not any(s.text for w in groups for s in w) and not frames:
        raise loaders.LoadError("no speech found" + ("" if figures.on else
                                                     " (set RAGGY_IMAGES=ocr or vision to read its frames)"))
    return render(groups, frames), None


def load_subtitles(data: bytes, options):
    from raggy.ingest import loaders

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise loaders.LoadError("not valid UTF-8 text") from None
    segs = parse_subtitles(text)
    if not segs:
        raise loaders.LoadError("no cues found")
    return render(windows(segs, options.media_window)), None
