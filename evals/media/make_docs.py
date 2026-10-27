"""Build the `media` eval suite: answers spoken in a podcast, shown in a video, or
captioned in a subtitle file (week 11).

    uv run python evals/media/make_docs.py      # macOS: uses `say` and ffmpeg

Two macOS voices read a scripted two-host podcast (Samantha as Emily, Daniel as
Paul). ffmpeg joins the lines into an .m4a, and builds an .mp4 from slide
pictures over a second narration. A third episode is written as captions only
(.vtt with voice tags), with exact cue times from the script. The outputs are
committed, so the suite doesn't depend on these tools at eval time.
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT = HERE / "docs"
VOICES = {"Emily": "Samantha", "Paul": "Daniel"}

PODCAST = [
    ("Emily", "Welcome to the Brightwater workshop podcast. I'm Emily."),
    ("Paul", "And I'm Paul. Today we're talking about looking after your cargo bike."),
    ("Emily", "Let's start with tyres. Check the pressure every two weeks."),
    ("Paul", "For the Ebb we recommend three bar in the rear tyre when the box is loaded."),
    ("Emily", "What about the chain?"),
    ("Paul", "Clean and oil the chain every three hundred kilometres, or after any ride in heavy rain."),
    ("Emily", "And brakes?"),
    ("Paul", "Replace the brake pads when the grooves are gone. A set costs twenty eight credits."),
    ("Emily", "Our winter service offer runs until the end of February."),
    ("Paul", "It includes a full brake check and a free chain for forty five credits."),
    ("Emily", "Thanks for listening. Next week, we visit the Fenwick workshop."),
]

VIDEO = [
    ("Ebb battery care", "Paul", "This video shows how to look after the Ebb battery."),
    ("Charging", "Paul", "Charge the battery indoors, at room temperature. A full charge takes four hours."),
    ("Storage", "Paul", "For winter storage, leave the battery at about sixty percent charge."),
    ("Warning light", "Paul", "If the red light blinks three times, the battery is too cold to charge."),
]

CAPTIONS = [  # (start, end, speaker, text): a livestream's published captions
    (2.0, 6.0, "Emily", "Welcome to the Ebb launch livestream."),
    (6.0, 11.0, "Paul", "The Ebb goes on sale on the twenty first of April."),
    (11.0, 16.0, "Emily", "Launch customers get a free rain cover for the cargo box."),
    (75.0, 81.0, "Paul", "Test rides start at the Calder market stall on launch weekend."),
    (81.0, 86.0, "Emily", "Book a test ride on our website."),
]


def font(size):
    try:
        return ImageFont.truetype("Helvetica", size)
    except OSError:
        return ImageFont.load_default(size)


def run(*args):
    subprocess.run(args, check=True, capture_output=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def speak(lines, tmp: Path, stem: str) -> list[Path]:
    paths = []
    for i, (who, text) in enumerate(lines):
        p = tmp / f"{stem}-{i:02d}.aiff"
        run("say", "-v", VOICES[who], "-o", str(p), text)
        paths.append(p)
    return paths


def concat(parts: list[Path], out: Path, tmp: Path) -> None:
    lst = tmp / f"{out.stem}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c:a", "aac", "-b:a", "64k", str(out))


def slide(title: str, text: str, path: Path) -> None:
    im = Image.new("RGB", (1280, 720), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 120], fill=(30, 80, 200))
    d.text((60, 35), title, fill="white", font=font(52))
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) > 40:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    for i, line in enumerate(lines):
        d.text((60, 200 + i * 70), line, fill="black", font=font(44))
    im.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        concat(speak(PODCAST, tmp, "pod"), OUT / "workshop-podcast-ep12.m4a", tmp)

        # The video: each section's slide shown for as long as its narration lasts.
        audio = speak([(who, text) for _, who, text in VIDEO], tmp, "vid")
        concat(audio, tmp / "narration.m4a", tmp)
        lst = tmp / "slides.txt"
        entries = []
        for i, ((title, _, text), a) in enumerate(zip(VIDEO, audio)):
            img = tmp / f"slide-{i}.png"
            slide(title, text, img)
            entries.append(f"file '{img}'\nduration {duration(a):.2f}\n")
        lst.write_text("".join(entries) + f"file '{tmp / f'slide-{len(VIDEO) - 1}.png'}'\n")
        run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-i", str(tmp / "narration.m4a"), "-vf", "fps=2,format=yuv420p", "-c:v", "libx264",
            "-c:a", "copy", "-shortest", str(OUT / "battery-care.mp4"))

    def stamp(s):
        return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"

    (OUT / "ebb-launch-livestream.vtt").write_text(
        "WEBVTT\n\n" + "".join(f"{stamp(a)} --> {stamp(b)}\n<v {who}>{text}\n\n" for a, b, who, text in CAPTIONS))
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
