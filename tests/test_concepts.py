from srag.types import Document
from srag.concepts import extract_concepts_from_text, build_concept_universe


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


def test_build_universe_top_l():
    docs = [
        Document(f"d{i}", f"document {i} about topic {i}", 1.0 - i * 0.1, 50)
        for i in range(10)
    ]
    universe = build_concept_universe(docs, top_l=3)
    assert universe.doc_concepts["d0"] != set()
    # Non-top-L docs still get concepts mapped (intersection with universe)
    # but the universe itself is only defined by top-L docs
    # Concepts only come from top-3 docs' vocabulary
    top3_concepts = set()
    for i in range(3):
        top3_concepts |= universe.doc_concepts[f"d{i}"]
    assert universe.concepts == top3_concepts
