from raggy.ingest.walk import iter_files

from .conftest import write


def names(root):
    return [p.relative_to(root).as_posix() for p in iter_files(root)]


def test_walk_respects_raggyignore(tmp_path):
    write(tmp_path / "keep.md", "k")
    write(tmp_path / "drafts" / "d.md", "d")
    write(tmp_path / "old.txt", "o")
    write(tmp_path / ".raggyignore", "drafts/\n*.txt\n")
    assert names(tmp_path) == ["keep.md"]


def test_walk_respects_gitignore_dirs(tmp_path):
    write(tmp_path / "src" / "a.md", "a")
    write(tmp_path / "build" / "b.md", "b")
    write(tmp_path / ".gitignore", "build\n")
    assert names(tmp_path) == ["src/a.md"]


def test_walk_skips_node_modules_and_hidden(tmp_path):
    write(tmp_path / "a.md", "a")
    write(tmp_path / "node_modules" / "pkg" / "README.md", "x")
    write(tmp_path / "__pycache__" / "x.txt", "x")
    write(tmp_path / ".venv" / "lib.md", "x")
    write(tmp_path / ".hidden.md", "x")
    assert names(tmp_path) == ["a.md"]


def test_walk_includes_html_and_pdf(tmp_path):
    for n in ("b.html", "a.pdf", "c.htm", "d.png", "e.markdown"):
        write(tmp_path / n, "x")
    # Week 10 reads images, so the walk includes them (loading one with images off is a named error).
    assert names(tmp_path) == ["a.pdf", "b.html", "c.htm", "d.png", "e.markdown"]
