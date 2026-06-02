"""Concept extraction and weighting from Appendix B.2 of the S-RAG paper."""

from collections.abc import Callable
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .types import ConceptUniverse, Document
from .utils import validate_candidates

_LEMMATIZER = WordNetLemmatizer()

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_VALID_TAGS = _NOUN_TAGS | _VERB_TAGS


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


def _penn_to_wordnet(tag: str) -> str | None:
    if tag in _NOUN_TAGS:
        return "n"
    if tag in _VERB_TAGS:
        return "v"
    return None


def extract_concepts_from_text(text: str) -> frozenset[str]:
    """Extract unique non-stopword noun/verb lemmas from a passage."""
    ignored = _stopwords()
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    concepts: set[str] = set()

    for word, tag in tagged:
        if tag not in _VALID_TAGS:
            continue
        if word in ignored or not word.isalpha() or len(word) < 2:
            continue
        lemma = _LEMMATIZER.lemmatize(word, pos=_penn_to_wordnet(tag))
        if lemma not in ignored:
            concepts.add(lemma)

    return frozenset(concepts)


def build_concept_universe(
    documents: list[Document],
    top_l: int = 20,
    extractor: Callable[[str], frozenset[str]] = extract_concepts_from_text,
) -> ConceptUniverse:
    """Construct U and w(u) following the paper's Appendix B.2 definition."""
    if top_l <= 0:
        raise ValueError("top_l must be positive")

    candidates = validate_candidates(documents)
    sorted_docs = sorted(
        candidates, key=lambda doc: (-doc.relevance_score, doc.doc_id)
    )
    extracted = {
        doc.doc_id: frozenset(extractor(doc.text)) for doc in sorted_docs
    }

    top_docs = sorted_docs[:top_l]
    all_concepts = frozenset().union(
        *(extracted[doc.doc_id] for doc in top_docs)
    )

    doc_concepts = {
        doc.doc_id: extracted[doc.doc_id] & all_concepts
        for doc in sorted_docs
    }
    concept_weights = {
        concept: max(
            doc.relevance_score
            for doc in sorted_docs
            if concept in doc_concepts[doc.doc_id]
        )
        for concept in all_concepts
    }

    return ConceptUniverse(
        concepts=all_concepts,
        concept_weights=concept_weights,
        doc_concepts=doc_concepts,
    )
