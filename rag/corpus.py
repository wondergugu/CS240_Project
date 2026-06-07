"""Independent retrieval corpus built from the KILT Wikipedia knowledge source.

The ELI5 corpus comes from KILT Wikipedia (not from the QA data under data/).
Passages are chunked and embedded once with bge; retrieval is a torch top-N
over the cached embeddings (no FAISS dependency). Corpus artifacts live under
`corpus/<dataset>/` (gitignored).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def chunk_words(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split a paragraph into overlapping word windows."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    words = cleaned.split()
    if len(words) <= size:
        return [cleaned]
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_kilt_corpus(
    out_dir: str | Path,
    max_pages: int = 5000,
    chunk_size: int = 120,
    min_chars: int = 80,
) -> tuple[int, Path]:
    """Stream the first `max_pages` KILT Wikipedia pages into passages.jsonl."""
    from datasets import load_dataset

    dataset = load_dataset(
        "facebook/kilt_wikipedia",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "passages.jsonl"

    written = 0
    try:
        with path.open("w") as handle:
            for index, page in enumerate(dataset):
                if index >= max_pages:
                    break
                wiki_id = page.get("wikipedia_id")
                title = page.get("wikipedia_title", "")
                text = page.get("text", {})
                paragraphs = text.get("paragraph", []) if isinstance(text, dict) else []
                for para_index, paragraph in enumerate(paragraphs):
                    if not isinstance(paragraph, str) or len(paragraph) < min_chars:
                        continue
                    for chunk_index, chunk in enumerate(chunk_words(paragraph, chunk_size)):
                        record = {
                            "pid": f"{wiki_id}:{para_index}:{chunk_index}",
                            "title": title,
                            "text": chunk,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
    except Exception as exc:  # noqa: BLE001 - keep passages streamed before a CDN error
        print(f"[warn] KILT stream stopped after {written} passages "
              f"({type(exc).__name__}); using partial corpus.")
    return written, path


def load_passages(corpus_dir: str | Path) -> list[dict]:
    """Read passages.jsonl back into memory."""
    path = Path(corpus_dir) / "passages.jsonl"
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TorchCorpusIndex:
    """In-memory dense index: embed passages once, retrieve top-N via cosine."""

    def __init__(
        self,
        passages: Sequence[dict],
        retriever,
        device: str | None = None,
        batch_size: int = 256,
    ):
        import torch

        self.passages = list(passages)
        self.retriever = retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = retriever.encode_passages(
            [p["text"] for p in self.passages], batch_size=batch_size
        )
        self.embeddings = torch.tensor(
            embeddings, dtype=torch.float16, device=self.device
        )

    def retrieve(self, query: str, top_n: int = 200) -> list[dict]:
        """Return the top-N passages (with cosine score) for a query."""
        import torch

        if not self.passages:
            return []
        query_vec = self.retriever.encode_queries([query])[0]
        query_tensor = torch.tensor(
            query_vec, dtype=torch.float16, device=self.device
        )
        scores = self.embeddings @ query_tensor
        k = min(top_n, len(self.passages))
        top = torch.topk(scores, k)
        return [
            {**self.passages[idx], "score": float(score)}
            for idx, score in zip(top.indices.tolist(), top.values.tolist())
        ]
