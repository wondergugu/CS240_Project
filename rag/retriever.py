"""Dense retriever wrapper around BAAI bge encoders.

Provides query/passage embeddings, relevance scoring `rel(d, q)`, and an
embedding cosine similarity that the MMR baseline can use in place of the
concept-Jaccard proxy. The model is open-weight and loaded locally.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# bge-*-en-v1.5 retrieval recommends prefixing queries with this instruction.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGERetriever:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        query_instruction: str = _QUERY_INSTRUCTION,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = query_instruction

    def encode_queries(self, queries: Sequence[str], batch_size: int = 64) -> np.ndarray:
        texts = [self.query_instruction + q for q in queries]
        return self._encode(texts, batch_size)

    def encode_passages(
        self, passages: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        return self._encode(list(passages), batch_size)

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Cosine relevance of each passage to the query (normalized embeddings)."""
        if not passages:
            return []
        q = self.encode_queries([query])[0]
        p = self.encode_passages(passages)
        return (p @ q).astype(float).tolist()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two (already L2-normalized) embeddings."""
    return float(np.dot(a, b))
