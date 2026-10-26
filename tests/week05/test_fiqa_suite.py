import hashlib
import io
import shutil
import zipfile
from pathlib import Path

from raggy.evals.suites import FIQA_SHA256, SUITES, download_beir, fiqa_suite, load_beir

FIXTURE = Path(__file__).resolve().parents[1] / "week02" / "fixtures" / "beir_tiny"


def test_load_beir_prefix():
    docs, qs = load_beir(FIXTURE, prefix="fiqa")
    assert all(d.source.startswith("fiqa:") for d in docs)
    assert all(g.source.startswith("fiqa:") for q in qs for g in q.gold)
    assert load_beir(FIXTURE)[0][0].source.startswith("scifact:")  # the default is unchanged


def test_fiqa_suite_from_local_copy(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "fiqa")
    s = fiqa_suite(tmp_path)
    assert s.name == "fiqa" and s.unit == "document" and s.ks == (1, 10, 100)
    assert s.dedupe is False and s.answers is False and s.fingerprint() == FIQA_SHA256
    assert s.documents()[0].source.startswith("fiqa:") and s.questions()
    assert "fiqa" in SUITES


def test_download_beir_generic(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in FIXTURE.rglob("*"):
            if f.is_file():
                z.write(f, "mini/" + f.relative_to(FIXTURE).as_posix())
    src = tmp_path / "mini.zip"
    src.write_bytes(buf.getvalue())
    folder = download_beir("mini", tmp_path / "data", src.as_uri(), hashlib.sha256(buf.getvalue()).hexdigest())
    assert (folder / "corpus.jsonl").exists() and folder.name == "mini"
