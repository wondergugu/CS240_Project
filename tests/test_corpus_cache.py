"""Tests for the on-disk passage-embedding cache in TorchCorpusIndex."""

import numpy as np
import torch

from rag.corpus import TorchCorpusIndex


class CountingRetriever:
    """Deterministic stub embedder that records how often it is called."""

    def __init__(self):
        self.calls = 0

    def encode_passages(self, texts, batch_size=256):
        self.calls += 1
        return np.array(
            [[float(i), float(len(t))] for i, t in enumerate(texts)], dtype=np.float32
        )


def _index(passages, retriever, cache_dir, key="model"):
    return TorchCorpusIndex(
        passages, retriever, device="cpu", cache_dir=cache_dir, cache_key=key
    )


def test_second_build_hits_cache(tmp_path):
    passages = [{"pid": "a", "text": "foo"}, {"pid": "b", "text": "barbar"}]
    r1 = CountingRetriever()
    idx1 = _index(passages, r1, tmp_path)
    assert r1.calls == 1  # first build embeds

    r2 = CountingRetriever()
    idx2 = _index(passages, r2, tmp_path)
    assert r2.calls == 0  # second build served from disk cache
    assert torch.equal(idx1.embeddings, idx2.embeddings)


def test_content_change_invalidates_cache(tmp_path):
    r1 = CountingRetriever()
    _index([{"pid": "a", "text": "foo"}], r1, tmp_path)
    assert r1.calls == 1

    r2 = CountingRetriever()
    _index([{"pid": "a", "text": "foo"}, {"pid": "c", "text": "new"}], r2, tmp_path)
    assert r2.calls == 1  # different ids/count -> recompute, not a stale hit


def test_different_model_key_does_not_collide(tmp_path):
    passages = [{"pid": "a", "text": "foo"}]
    _index(passages, CountingRetriever(), tmp_path, key="bge-large")
    r = CountingRetriever()
    _index(passages, r, tmp_path, key="bge-small")
    assert r.calls == 1  # a different retriever key recomputes its own cache


def test_no_cache_dir_keeps_original_behavior(tmp_path):
    passages = [{"pid": "a", "text": "foo"}]
    r = CountingRetriever()
    TorchCorpusIndex(passages, r, device="cpu")  # cache_dir omitted
    r2 = CountingRetriever()
    TorchCorpusIndex(passages, r2, device="cpu")
    assert r.calls == 1 and r2.calls == 1  # always recompute, no files written
    assert not list(tmp_path.glob("*.npy"))
