from raggy.ingest.registry import Record, Registry


def rec(source, **kw):
    base = dict(source=source, kind="file", content_hash="h", size=1, mtime_ns=2, pipeline_key="k",
                signature=None, duplicate_of=None, title="t", doc_type="markdown", modified=None,
                chunks=3, ingested_at="2026-10-25T00:00:00Z")
    return Record(**(base | kw))


def test_registry_roundtrip(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    r.upsert(rec("/a/x.md", signature=b"\x01\x02"))
    r.commit()
    again = Registry(tmp_path / "reg.sqlite")
    assert again.get("/a/x.md") == rec("/a/x.md", signature=b"\x01\x02")
    assert [x.source for x in again.all()] == ["/a/x.md"]


def test_uncommitted_writes_discarded(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    r.upsert(rec("/a/x.md"))
    r.close()
    assert Registry(tmp_path / "reg.sqlite").get("/a/x.md") is None


def test_under_does_not_match_sibling_prefix(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    for s in ("/a/doc/1.md", "/a/doc/sub/2.md", "/a/docs2/3.md"):
        r.upsert(rec(s))
    r.upsert(rec("https://x.example/a/doc/4", kind="url"))
    assert sorted(x.source for x in r.under("/a/doc")) == ["/a/doc/1.md", "/a/doc/sub/2.md"]


def test_upsert_replaces_and_delete(tmp_path):
    r = Registry(tmp_path / "reg.sqlite")
    r.upsert(rec("/x"))
    r.upsert(rec("/x", chunks=9))
    assert r.get("/x").chunks == 9
    r.upsert(rec("/y", duplicate_of="/x"))
    assert [d.source for d in r.duplicates_of("/x")] == ["/y"]
    r.delete("/x")
    assert r.get("/x") is None
