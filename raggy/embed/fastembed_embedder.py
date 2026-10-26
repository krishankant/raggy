"""A real embedding model running locally through fastembed (ONNX, no PyTorch).

The default, BAAI/bge-small-en-v1.5, is a 67 MB model with 384 dimensions. It
downloads once on first use and then runs offline on a laptop CPU.
"""

from __future__ import annotations

import numpy as np

from raggy.embed.base import normalize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16):
        from fastembed import TextEmbedding  # imported here so tests never load it

        self.name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)
        self.dim = self._model.embedding_size
        self._tokenizer = None
        self.max_tokens = 510  # bge-small: 512 including [CLS] and [SEP]

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            # A copy without truncation: the model's own tokenizer stops counting at 512.
            self._tokenizer = Tokenizer.from_str(self._model.model.tokenizer.to_str())
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # A batch is padded to its longest text, so mixing a 20-token title with a
        # 250-token chunk makes the model compute 230 tokens of padding. Embedding
        # shortest-first keeps batches even; the input order is restored after.
        # (Measured on an 8 GB M2: batch 16 beats batch 64, and worker processes
        # are slower still, because each loads its own copy of the model.)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        # passage_embed adds the model's "document" prefix when it has one.
        vectors = list(self._model.passage_embed([texts[i] for i in order], batch_size=self.batch_size))
        if not vectors:
            return np.zeros((0, self.dim), np.float32)
        out = np.empty((len(texts), len(vectors[0])), np.float32)
        out[order] = np.stack(vectors)
        return normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the "query" prefix (bge: "Represent this sentence…").
        return normalize(next(iter(self._model.query_embed(text))))
