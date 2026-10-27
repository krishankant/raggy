# Week 11: Audio and Video — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make speech and video searchable and citable by time. Audio and video files, subtitle files and remote media URLs become time-labelled text, and each passage is cited as `file @ mm:ss–mm:ss`.

**Architecture:**
- A `Transcriber` (faster-whisper, local, CPU int8, cached by audio bytes + model + version) turns media into `Segment(start, end, text, speaker)`.
- `media.windows()` groups segments into windows of at most `RAGGY_MEDIA_WINDOW` seconds (60), starting a new window at every speaker change.
- Each window becomes a labelled part (`@ 01:23`), so week 9's page-label machinery gives every chunk its time range for free.
- Video adds keyframes: one frame every `RAGGY_KEYFRAME_SECONDS` (30), skipping near-identical frames. Each is read by week 10's image reader and placed in the window covering its time.
- Subtitle files (`.srt`, `.vtt`, with `<v Speaker>` voice tags) load directly, with no model.
- Remote media (YouTube and others) goes through yt-dlp: published captions when there are any, otherwise the audio, which is then transcribed.

**Tech Stack:** faster-whisper 1.2.1 (CTranslate2; bundles PyAV 19 for decoding, so no system ffmpeg). Whisper `base` is the default (about 145 MB); `small`, `medium` and `large-v3` are the larger options (spec section 9). yt-dlp 2026.10.26 is an optional extra (`uv sync --extra media`).

**Spec:** `docs/superpowers/specs/2026-10-23-raggy-course-design.md` (section 7, week 11 row).

## Global Constraints

- Tests are offline: a fake transcriber, and tiny videos encoded in the test with PyAV's `mpeg4` codec.
- `RAGGY_MEDIA=off|on`, default `off` until measured. With it off, a media file is a named `LoadError` ("set RAGGY_MEDIA=on"). Subtitle files always load.
- `RAGGY_WHISPER_MODEL` (`base`), `RAGGY_MEDIA_WINDOW` (60), `RAGGY_KEYFRAME_SECONDS` (30, video, needs `RAGGY_IMAGES` on). All of them, plus `TRANSCRIBE_VERSION`, go in `pipeline_key` and the eval key.
- Times print as `mm:ss`, or `h:mm:ss` past an hour.
- Speaker turns come only from sources that carry them (subtitle voice tags, `Name:` line prefixes). Automatic diarization is out of scope: it needs pyannote and a gated model. The lesson names the gap.
- **Decision rule, set before any run:** `RAGGY_MEDIA=on` becomes the default if the `media` suite's recall@4 under `on` rises against `off` (where every answer is missing), the earlier suites stay identical, and `base` transcribes faster than real time on this M2. Not measured for this release, so it stays off.

## Review Focus

1. A silent or zero-length file: an empty transcript is a `LoadError` ("no speech found"), not an empty document.
2. A segment longer than the window: it isn't split mid-sentence; its window just runs long.
3. Keyframes in a two-hour video: the frame step caps at 120 frames per file (the step grows), so ingest stays bounded.
4. A URL yt-dlp can't handle, or yt-dlp not installed: a `LoadError` with the hint, never a crash.
5. A subtitle file with HTML tags or duplicated rolling-caption lines: tags are stripped and consecutive repeated lines are dropped.

### Task 1: transcription, windows, subtitles, the media loader (TDD)
### Task 2: video keyframes, remote media, CLI and keys (TDD)
### Task 3: the `media` suite generator (macOS `say` + ffmpeg), lesson, guide, docs; release
