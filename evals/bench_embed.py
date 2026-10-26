"""Embedding throughput: the same texts, in input order vs sorted by length (week 5).

    uv run python evals/bench_embed.py evals/data/fiqa 800
"""

import random
import sys
import time
from pathlib import Path


def main() -> None:
    from fastembed import TextEmbedding

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from raggy.evals.suites import load_beir

    folder, n = Path(sys.argv[1]), int(sys.argv[2])
    docs, _ = load_beir(folder, prefix=folder.name)
    texts = [d.text[:1000] for d in random.Random(0).sample(docs, n)]
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    list(model.passage_embed(texts[:32], batch_size=16))  # warm up
    for name, batch in (("input order", texts), ("sorted by length", sorted(texts, key=len))):
        t = time.perf_counter()
        list(model.passage_embed(batch, batch_size=16))
        print(f"{name:18} {n / (time.perf_counter() - t):6.1f} texts/s", flush=True)


if __name__ == "__main__":
    main()
