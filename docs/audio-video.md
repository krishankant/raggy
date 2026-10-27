# Audio and video: from speech to citable text

**[← Docs hub](README.md)** · **[Week 11 lesson](../lessons/week11-audio-video.md)**

This guide sits beside week 11. It explains how a speech model produces timed text, how raggy groups that text into retrievable windows that keep their times, how video frames join the transcript, and what to do about speakers.

---

## 1. What Whisper produces

Whisper is a sequence-to-sequence model: audio goes in as a log-mel spectrogram (a picture of which frequencies are loud over time), and text comes out, token by token. It works on 30-second windows. Interleaved with the words, it predicts timestamp tokens, so the output is a list of segments:

```text
[00:24.0 → 00:31.5]  For the Ebb we recommend three bar in the rear tyre when the box is loaded.
[00:31.5 → 00:36.2]  What about the chain?
```

For long recordings, the windows are stitched: each one is decoded with the previous window's text as context, and the next window starts where the last segment ended.

Three failure modes matter for RAG:

- **Invented text over silence or music.** The model is a language model and will produce plausible words when there are none. A voice-activity detector (VAD) that removes non-speech first is the standard fix, and raggy turns it on.
- **Names and jargon.** "Brightwater" can come out as "bright water", or a product code as nonsense. Larger models help, and so does a short prompt of expected words (an option faster-whisper has, not used here).
- **Numbers.** "Three hundred kilometres" may be written "300 km". Search should tolerate both; that's one more reason to use hybrid search (week 6) on transcripts.

## 2. From segments to windows to chunks

Segments are a few seconds each: too small to retrieve, and they cut sentences mid-thought. raggy groups them:

1. **Windows.** Consecutive segments join one window while they share a speaker and the window stays within `RAGGY_MEDIA_WINDOW` seconds (60).
2. **Labels.** Each window becomes one labelled part, `("@ 00:24", text)`, exactly like week 9's pages and slides.
3. **Chunks.** Week 4's chunker splits the document; week 9 maps each chunk's offsets back to labels. A chunk inside one window cites `@ 00:24`; one spanning windows cites `@ 00:24–01:30`.

```text
segments:  |--E--|--E--|---P---|----P----|--E--|
windows:   [ E 00:00 ][ P 00:14          ][E 01:20]   ← a new window at each speaker change
chunks:    [-----chunk 1-----][-----chunk 2------]
citations:  @ 00:00–00:14       @ 00:14–01:20
```

Why a speaker change always starts a window: if Emily asks "Is the battery covered for three years?" and Paul answers "No, two", a chunk that blends them under no speaker reads like a claim of three years.

## 3. Video

A video is two streams. raggy reads both:

- **Audio:** transcribed as above.
- **Frames:** one every `RAGGY_KEYFRAME_SECONDS` (30), read by week 10's image reader (OCR or a vision model).

Sampling every 30 s would read a static slide forty times in a 20-minute talk. So each sampled frame is shrunk to a 16 × 16 grey thumbnail and compared with the last *kept* one; if the mean pixel difference is under 6 (of 255), it's skipped. Slide changes, cuts and new scenes pass; the same slide doesn't. For a long video, the step grows so that at most 120 frames are read.

Each kept frame becomes `[Image: talk.mp4 @ 02:30, frame]` in the window covering 02:30. A question about what was *shown* ("what does the warning light look like?") can then be answered from the frame's text, and cited by time.

## 4. Speakers

Who said it is part of the meaning. raggy takes speaker names from sources that have them:

| Source | Example |
| --- | --- |
| WebVTT voice tags | `<v Emily>Launch day is 21 April.` |
| Name prefixes | `Paul: Thanks, Emily.` |

Whisper doesn't identify speakers. Diarization ("who spoke when") is a separate model: typically pyannote, which segments audio by voice and clusters the segments into speakers ("SPEAKER_00", "SPEAKER_01"). To add it, run diarization over the same audio, give each Whisper segment the speaker whose turn overlaps it most, and pass the segments on with `speaker` set. `windows()` then splits on turns unchanged. It's out of scope here because pyannote's models need an account and a licence acceptance, and they add a second heavy model.

## 5. Captions first

If the media already has captions, use them:

- Captions written by a person are exact, and often name speakers.
- They're already timed.
- They cost nothing: no model, no download beyond a few kilobytes.

raggy's subtitle loader reads `.srt` and `.vtt` directly, strips formatting tags, and drops the repeated lines of rolling captions (where each cue repeats the previous line plus a new one). For a media page, it asks yt-dlp for published English captions before downloading any audio. Automatic captions are skipped on purpose: they're another speech model's output, usually worse than Whisper `small`.

## 6. Costs

| Step | Cost |
| --- | --- |
| Download a Whisper model | Once: 145 MB (`base`) to 3 GB (`large-v3`) |
| Transcribe | CPU time, a fraction of the audio's length for `base`, more for larger models; once per file (cached by bytes + model) |
| Keyframes | Decode + one image read per kept frame (week 10's costs) |
| Captions | Effectively free |

Measure transcription speed on your own machine with the `media` suite before turning `RAGGY_MEDIA=on` for a large archive.

## Further reading

- [Whisper](https://arxiv.org/abs/2212.04356), Radford et al. (2022).
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): the CTranslate2 implementation used here; the README has speed and memory tables.
- [pyannote.audio](https://github.com/pyannote/pyannote-audio): speaker diarization.
- [WebVTT](https://www.w3.org/TR/webvtt1/): the caption format, including voice tags.
