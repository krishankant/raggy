"""Search latency at the depth `raggy ask` uses: numpy exact vs LanceDB exact vs LanceDB ANN (week 5).

Reads the vectors of an eval index built with RAGGY_ANN=on (so the ANN index exists),
loads the same rows into a NumpyStore, and times store.search for the suite's questions.

    RAGGY_ANN=on uv run raggy eval --suite fiqa --retrieval-only     # builds .raggy/evals/fiqa
    uv run python evals/bench_search.py .raggy/evals/fiqa/lance evals/data/fiqa 300 16
"""

import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from raggy.embed.fastembed_embedder import FastEmbedEmbedder
    from raggy.evals.suites import load_beir
    from raggy.store.lance_store import LanceStore
    from raggy.store.numpy_store import NumpyStore

    lance_dir, data, n, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    lance = LanceStore(lance_dir)
    table = lance._tbl.to_arrow()
    num = NumpyStore(lance_dir / "unused")
    num.embedder = lance.embedder
    num.vectors = np.stack(table.column("vector").to_numpy(zero_copy_only=False)).astype(np.float32)
    num.chunks = [lance._chunk(r) for r in table.drop_columns(["vector"]).to_pylist()]
    _, questions = load_beir(data, prefix=data.name)
    emb = FastEmbedEmbedder()
    vecs = [emb.embed_query(q.question) for q in questions[:n]]

    def bench(fn):
        fn(vecs[0])
        times = []
        for v in vecs:
            t = time.perf_counter()
            fn(v)
            times.append((time.perf_counter() - t) * 1000)
        return np.percentile(times, 50), np.percentile(times, 95)

    for name, fn in (("numpy exact", lambda v: num.search(v, k, emb.name)),
                     ("lance exact", lambda v: lance.search(v, k, emb.name, exact=True)),
                     ("lance ann", lambda v: lance.search(v, k, emb.name, exact=False))):
        p50, p95 = bench(fn)
        print(f"{name:12} p50 {p50:6.2f} ms  p95 {p95:6.2f} ms")
    overlap = np.mean([len({h.chunk.id for h in lance.search(v, k, emb.name, exact=False)}
                           & {h.chunk.id for h in num.search(v, k, emb.name)}) / k for v in vecs])
    print(f"ann overlap@{k} vs exact: {overlap:.3f}  ({len(num.chunks)} chunks, {len(vecs)} questions)")


if __name__ == "__main__":
    main()
