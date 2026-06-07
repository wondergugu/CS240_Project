"""Independent retrieval corpus built from the KILT Wikipedia knowledge source.

The ELI5 corpus comes from KILT Wikipedia (not from the QA data under data/).
The sist corpus is local: it is built from `data/sist/jsonl/chunks.jsonl` (the
pre-chunked SIST knowledge base) and belongs exclusively to the sist dataset.
Passages are embedded once with bge; retrieval is a torch top-N over the
embeddings (no FAISS dependency). For larger corpora (sist has ~23k chunks),
`TorchCorpusIndex` can cache the passage embeddings to disk so they are not
recomputed on every run. Corpus artifacts live under `corpus/<dataset>/`
(gitignored).
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


# SIST page chunks carry a repeated site navigation menu; these exact short
# menu labels (as standalone lines) are dropped so they do not pollute the
# passage text. Real body content rarely appears as a bare one-line menu label.
_SIST_NAV_LINES = frozenset(
    {
        "English", "学校首页", "导航", "首 页", "首页", "学院概况",
        "使命和愿景", "学院介绍", "院长寄语", "顾问委员会", "院务委员会",
        "科学研究", "研究中心", "联合实验室", "省部级研究中心", "规章制度",
        "采购公告", "常用资料", "资料下载", "人才培养", "招生就业",
        "新闻动态", "通知公告", "学院首页",
    }
)


def _clean_sist_text(text: str) -> str:
    """Drop bare navigation-menu lines and normalize whitespace for a chunk."""
    kept = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and line.strip() not in _SIST_NAV_LINES
    ]
    return " ".join(" ".join(kept).split())


def build_sist_corpus(
    out_dir: str | Path,
    data_dir: str | Path | None = None,
    min_chars: int = 80,
) -> tuple[int, Path]:
    """Build the sist passages.jsonl from data/sist's pre-chunked knowledge base.

    Reads `<data_dir>/jsonl/chunks.jsonl` (one chunk per line) and writes records
    in the same `{pid, title, text}` schema the ELI5 corpus uses, plus `url` and
    `category` for traceability. Chunks whose cleaned text is shorter than
    `min_chars` are dropped (this clears the handful of near-empty chunks). The
    corpus is local-only — no network, no KILT — and is exclusive to sist.
    """
    if data_dir is None:
        from .config import DATA_DIR

        data_dir = Path(DATA_DIR) / "sist"
    chunks_path = Path(data_dir) / "jsonl" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"no sist chunks at {chunks_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "passages.jsonl"

    written = 0
    with chunks_path.open(encoding="utf-8") as src, path.open("w") as handle:
        for line in src:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            text = _clean_sist_text(chunk.get("text") or "")
            if len(text) < min_chars:
                continue
            record = {
                "pid": f"sist:{chunk.get('id')}",
                "title": chunk.get("title") or "",
                "text": text,
                "url": chunk.get("url") or "",
                "category": chunk.get("category") or "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
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
        cache_dir: str | Path | None = None,
        cache_key: str = "",
    ):
        import numpy as np
        import torch

        self.passages = list(passages)
        self.retriever = retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = self._load_or_build_embeddings(
            np, batch_size, cache_dir, cache_key
        )
        self.embeddings = torch.tensor(
            embeddings, dtype=torch.float16, device=self.device
        )

    def _fingerprint(self) -> str:
        """Short hash of the passage ids; changes if the corpus content changes."""
        import hashlib

        digest = hashlib.sha1()
        for passage in self.passages:
            digest.update(str(passage.get("pid", "")).encode())
            digest.update(b"\x00")
        return digest.hexdigest()[:8]

    def _cache_path(self, cache_dir: str | Path, cache_key: str) -> Path:
        slug = "".join(c if c.isalnum() else "-" for c in str(cache_key))
        name = f"emb_{slug}_{len(self.passages)}_{self._fingerprint()}.npy"
        return Path(cache_dir) / name

    def _load_or_build_embeddings(self, np, batch_size, cache_dir, cache_key):
        """Reuse cached passage embeddings when valid, else embed and cache them.

        The cache file is keyed by (retriever model, passage count, id
        fingerprint), so a corpus rebuild or a retriever swap invalidates it
        automatically. With no cache_dir the behavior is the original (always
        recompute), keeping callers that omit it unchanged.
        """
        path = None
        if cache_dir is not None and self.passages:
            path = self._cache_path(cache_dir, cache_key)
            if path.exists():
                try:
                    cached = np.load(path)
                    if cached.shape[0] == len(self.passages):
                        return cached
                except Exception:  # noqa: BLE001 - a bad cache must not abort
                    pass
        embeddings = self.retriever.encode_passages(
            [p["text"] for p in self.passages], batch_size=batch_size
        )
        embeddings = np.asarray(embeddings, dtype=np.float16)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, embeddings)
            except Exception:  # noqa: BLE001 - caching is best-effort
                pass
        return embeddings

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
