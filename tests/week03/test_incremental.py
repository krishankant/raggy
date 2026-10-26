import dataclasses
import random

import httpx
import pytest

import raggy.ingest.fetch as fetch_mod
import raggy.pipeline as pipeline_mod
from raggy.config import Config
from raggy.embed.fake import FakeEmbedder
from raggy.generate.fake import FakeLLM
from raggy.ingest.registry import Registry
from raggy.pipeline import Raggy
from raggy.store.numpy_store import NumpyStore

from .conftest import make_pdf, write

VOCAB = [f"w{i}" for i in range(3000)]


def words(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) for _ in range(n))


def make(tmp_path, **cfg) -> Raggy:
    c = Config(index_dir=tmp_path / "index", embed_model="fake", llm_provider="fake", llm_model=None,
               **({"store": "numpy"} | cfg))
    return Raggy(c, embedder=FakeEmbedder(), llm=FakeLLM())


def stored(tmp_path):
    return {k.rsplit("/", 1)[-1]: v for k, v in NumpyStore.load(tmp_path / "index").sources().items()}


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    write(d / "a.md", "# Alpha\n\nRefunds are paid within 14 days.")
    write(d / "b.md", "# Beta\n\nThe office cat is called Biscuit.")
    return d


def test_first_ingest_adds(tmp_path, docs):
    r = make(tmp_path).ingest([str(docs)])
    assert (r.added, r.updated, r.unchanged, r.removed, r.documents) == (2, 0, 0, 0, 2)
    assert stored(tmp_path) == {"a.md": 1, "b.md": 1}
    rec = Registry(tmp_path / "index" / "registry.sqlite").get(str((docs / "a.md").resolve()))
    assert rec.title == "Alpha" and rec.chunks == 1 and rec.doc_type == "markdown"


def test_second_ingest_skips_unchanged_without_reading(tmp_path, docs, monkeypatch):
    make(tmp_path).ingest([str(docs)])

    def boom(*a, **k):
        raise AssertionError("an unchanged file was read")
    monkeypatch.setattr(pipeline_mod, "load_file", boom)
    r = make(tmp_path).ingest([str(docs)])
    assert (r.added, r.updated, r.unchanged, r.chunks) == (0, 0, 2, 0)


def test_touched_but_same_content_is_unchanged(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    (docs / "a.md").write_text((docs / "a.md").read_text())  # new mtime, same bytes
    r = make(tmp_path).ingest([str(docs)])
    assert (r.unchanged, r.updated, r.chunks) == (2, 0, 0)


def test_changed_file_reindexed(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    write(docs / "a.md", "# Alpha\n\nRefunds now take 30 days.")
    r = make(tmp_path).ingest([str(docs)])
    assert (r.updated, r.unchanged) == (1, 1)
    hits = make(tmp_path).search("refunds days", k=1)
    assert "30 days" in hits[0].chunk.text


def test_deleted_file_removed(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    (docs / "b.md").unlink()
    r = make(tmp_path).ingest([str(docs)])
    assert r.removed == 1 and stored(tmp_path) == {"a.md": 1}
    assert Registry(tmp_path / "index" / "registry.sqlite").get(str((docs / "b.md").resolve())) is None
    assert (docs / "a.md").exists()


def test_newly_ignored_file_removed(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    write(docs / ".raggyignore", "b.md\n")
    assert make(tmp_path).ingest([str(docs)]).removed == 1
    assert (docs / "b.md").exists()


def test_missing_folder_target_keeps_its_sources(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    moved = tmp_path / "elsewhere"
    docs.rename(moved)
    r = make(tmp_path).ingest([str(docs)])
    assert r.skipped == [str(docs)] and r.removed == 0
    assert stored(tmp_path) == {"a.md": 1, "b.md": 1}


def test_settings_change_reindexes_unchanged_files(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    r = make(tmp_path, chunk_size=400, chunk_overlap=50).ingest([str(docs)])
    assert (r.updated, r.unchanged) == (2, 0)
    r = make(tmp_path, chunk_size=400, chunk_overlap=50, html_extract="full").ingest([str(docs)])
    assert r.updated == 2


def test_registry_commit_after_store_save(tmp_path, docs, monkeypatch):
    def fail(self):
        raise RuntimeError("disk full")
    with monkeypatch.context() as m:
        m.setattr(Registry, "commit", fail)
        with pytest.raises(RuntimeError):
            make(tmp_path).ingest([str(docs)])
    assert stored(tmp_path) == {"a.md": 1, "b.md": 1}  # the store was saved first
    r = make(tmp_path).ingest([str(docs)])
    assert r.unchanged == 0 and r.updated + r.added == 2  # re-indexed, not trusted


def test_corrupt_pdf_is_reported_and_run_continues(tmp_path, docs):
    write(docs / "broken.pdf", b"%PDF-1.4 garbage")
    write(docs / "ok.pdf", make_pdf(["Tyre pressure is 4 bar."]))
    r = make(tmp_path).ingest([str(docs)])
    assert r.added == 3 and [s.rsplit("/", 1)[-1] for s, _ in r.failed] == ["broken.pdf"]
    assert "not a readable PDF" in r.failed[0][1]


def test_previously_indexed_file_that_breaks_keeps_old_chunks(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    (docs / "a.md").write_bytes(b"\xff\xfe broken \x81")
    r = make(tmp_path).ingest([str(docs)])
    assert len(r.failed) == 1 and r.removed == 0
    assert stored(tmp_path) == {"a.md": 1, "b.md": 1}


def url_client(pages):
    def handler(req):
        body = pages[str(req.url)]
        if isinstance(body, int):
            return httpx.Response(body)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=body.encode())
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_url_ingest_and_unchanged_refetch(tmp_path, monkeypatch):
    pages = {"https://x.example/faq": "Opening hours are nine to five."}
    real = fetch_mod.fetch_url
    monkeypatch.setattr(pipeline_mod, "fetch_url",
                        lambda url, options=None: real(url, options, client=url_client(pages)))
    assert make(tmp_path).ingest(["https://x.example/faq"]).added == 1
    assert make(tmp_path).ingest(["https://x.example/faq"]).unchanged == 1
    pages["https://x.example/faq"] = "Opening hours are ten to six."
    assert make(tmp_path).ingest(["https://x.example/faq"]).updated == 1


def test_failed_refetch_keeps_old_chunks(tmp_path, monkeypatch):
    pages = {"https://x.example/faq": "Opening hours are nine to five."}
    real = fetch_mod.fetch_url
    monkeypatch.setattr(pipeline_mod, "fetch_url",
                        lambda url, options=None: real(url, options, client=url_client(pages)))
    make(tmp_path).ingest(["https://x.example/faq"])
    pages["https://x.example/faq"] = 503
    r = make(tmp_path).ingest(["https://x.example/faq"])
    assert "503" in r.failed[0][1] and r.removed == 0
    assert NumpyStore.load(tmp_path / "index").sources() == {"https://x.example/faq": 1}


def test_duplicate_recorded_not_embedded(tmp_path, docs):
    write(docs / "archive" / "a-copy.md", (docs / "a.md").read_text())
    r = make(tmp_path).ingest([str(docs)])
    assert (r.added, r.duplicates) == (2, 1)
    assert stored(tmp_path) == {"a.md": 1, "b.md": 1}
    rec = Registry(tmp_path / "index" / "registry.sqlite").get(str((docs / "archive" / "a-copy.md").resolve()))
    assert rec.duplicate_of == str((docs / "a.md").resolve()) and rec.chunks == 0


def test_near_duplicate_detected_on_ingest(tmp_path, docs):
    body = words(200, 1)
    write(docs / "long.md", body)
    write(docs / "archive" / "long-old.md", body.replace(body.split()[50], "older", 1))
    assert make(tmp_path).ingest([str(docs)]).duplicates == 1


def test_dedupe_off_indexes_copies(tmp_path, docs):
    write(docs / "archive" / "a-copy.md", (docs / "a.md").read_text())
    r = make(tmp_path, dedupe=False).ingest([str(docs)])
    assert (r.added, r.duplicates) == (3, 0)


def test_duplicate_promoted_when_canonical_deleted(tmp_path, docs):
    write(docs / "archive" / "a-copy.md", (docs / "a.md").read_text())
    make(tmp_path).ingest([str(docs)])
    (docs / "a.md").unlink()
    r = make(tmp_path).ingest([str(docs)])
    assert r.removed == 1 and r.duplicates == 0
    assert stored(tmp_path) == {"a-copy.md": 1, "b.md": 1}


def test_duplicate_promoted_when_canonical_changes(tmp_path, docs):
    write(docs / "archive" / "a-copy.md", (docs / "a.md").read_text())
    make(tmp_path).ingest([str(docs)])
    write(docs / "a.md", "# Alpha\n\nSomething entirely different now.")
    make(tmp_path).ingest([str(docs)])
    assert stored(tmp_path) == {"a.md": 1, "a-copy.md": 1, "b.md": 1}


def test_duplicate_promoted_after_raggy_remove(tmp_path, docs):
    write(docs / "archive" / "a-copy.md", (docs / "a.md").read_text())
    make(tmp_path).ingest([str(docs)])
    make(tmp_path).remove(str(docs / "a.md"))
    make(tmp_path).ingest([str(docs / "archive")])
    assert "a-copy.md" in stored(tmp_path)


def test_full_flag_rehashes(tmp_path, docs, monkeypatch):
    make(tmp_path).ingest([str(docs)])
    calls = []
    real = pipeline_mod.load_file
    monkeypatch.setattr(pipeline_mod, "load_file", lambda p, o=None: calls.append(p) or real(p, o))
    r = make(tmp_path).ingest([str(docs)], full=True)
    assert len(calls) == 2 and r.unchanged == 2


def test_remove_folder_prefix(tmp_path, docs):
    write(tmp_path / "docs2" / "c.md", "Another folder entirely.")
    make(tmp_path).ingest([str(docs), str(tmp_path / "docs2")])
    assert make(tmp_path).remove(str(docs)) == 2
    assert stored(tmp_path) == {"c.md": 1}
    assert (docs / "a.md").exists()


def test_unchanged_run_does_not_load_embedder(tmp_path, docs):
    make(tmp_path).ingest([str(docs)])
    c = Config(index_dir=tmp_path / "index", embed_model="fake-bow-64", llm_provider="fake", llm_model=None,
               store="numpy")
    r = Raggy(c)  # no embedder injected: touching .embedder would try to load a real model
    assert r.ingest([str(docs)]).unchanged == 2 and r._embedder is None


def test_old_index_without_registry_is_adopted(tmp_path, docs):
    make(tmp_path).ingest_documents(pipeline_mod.load_path(docs))  # week-2 style, no registry
    r = make(tmp_path).ingest([str(docs)])
    assert r.added == 2 and stored(tmp_path) == {"a.md": 1, "b.md": 1}
