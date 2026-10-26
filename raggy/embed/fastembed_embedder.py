"""A real embedding model running locally through fastembed (ONNX, no PyTorch).

The default, BAAI/bge-small-en-v1.5, is a 67 MB model with 384 dimensions. It
downloads once on first use and then runs offline on a laptop CPU.
"""

from __future__ import annotations

import numpy as np

from raggy.embed.base import normalize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
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
        # passage_embed adds the model's "document" prefix when it has one.
        vectors = list(self._model.passage_embed(texts, batch_size=self.batch_size))
        return normalize(np.stack(vectors)) if vectors else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the "query" prefix (bge: "Represent this sentence…").
        return normalize(next(iter(self._model.query_embed(text))))
