import pytest

from srag.concepts import build_concept_universe, extract_concepts_from_text
from srag.types import Document


def test_extract_basic():
    text = "The quick brown fox jumps over the lazy dog"
    concepts = extract_concepts_from_text(text)
    assert "fox" in concepts
    assert "dog" in concepts
    assert "jump" in concepts
    assert "the" not in concepts
    assert "over" not in concepts


def test_extract_filters_stopwords():
    text = "This is a very important thing to do"
    concepts = extract_concepts_from_text(text)
    assert "thing" in concepts
    assert "is" not in concepts
    assert "a" not in concepts


def test_extract_lemmatizes():
    text = "The dogs were running and the cats were sleeping"
    concepts = extract_concepts_from_text(text)
    assert "dog" in concepts
    assert "cat" in concepts
    assert "run" in concepts
    assert "sleep" in concepts


def test_build_universe_weights():
    docs = [
        Document("d0", "climate change causes global warming", 0.9, 50),
        Document("d1", "climate change effects on ecosystems", 0.7, 60),
        Document("d2", "renewable energy solutions reduce emissions", 0.5, 55),
    ]
    universe = build_concept_universe(docs, top_l=3)
    assert "climate" in universe.concepts or "change" in universe.concepts
    if "climate" in universe.concepts:
        assert universe.concept_weights["climate"] == 0.9


def test_build_universe_extracts_each_document_once_and_maps_all_documents():
    docs = [
        Document("anchor", "anchor", 0.9, 10),
        Document("candidate", "candidate", 0.5, 10),
    ]
    extractions: list[str] = []
    mapped = {
        "anchor": frozenset({"shared"}),
        "candidate": frozenset({"shared", "outside"}),
    }

    def extractor(text: str) -> frozenset[str]:
        extractions.append(text)
        return mapped[text]

    universe = build_concept_universe(docs, top_l=1, extractor=extractor)

    assert extractions == ["anchor", "candidate"]
    assert universe.concepts == {"shared"}
    assert universe.doc_concepts["candidate"] == {"shared"}
    assert universe.concept_weights["shared"] == 0.9


def test_build_universe_top_l():
    docs = [
        Document(f"d{i}", f"document {i} about topic {i}", 1.0 - i * 0.1, 50)
        for i in range(10)
    ]
    universe = build_concept_universe(docs, top_l=3)
    assert universe.doc_concepts["d0"]
    top3_concepts = set()
    for i in range(3):
        top3_concepts |= universe.doc_concepts[f"d{i}"]
    assert universe.concepts == top3_concepts


def test_build_universe_rejects_invalid_configuration():
    doc = Document("d0", "content", 1.0, 10)
    with pytest.raises(ValueError, match="top_l"):
        build_concept_universe([doc], top_l=0)

    with pytest.raises(ValueError, match="duplicate document id"):
        build_concept_universe([doc, doc], top_l=1)
