"""Concept extraction pipeline (Section B.2 of the paper).

Extracts noun/verb lemmas from documents to build the concept universe U.
Concept weights: w(u) = max_{d: u in U(d)} rel(d, q)
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from .types import Document, ConceptUniverse

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_VALID_TAGS = _NOUN_TAGS | _VERB_TAGS


def _penn_to_wordnet(tag: str):
    if tag in _NOUN_TAGS:
        return wordnet.NOUN
    if tag in _VERB_TAGS:
        return wordnet.VERB
    return None


def extract_concepts_from_text(text: str) -> set[str]:
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    concepts = set()
    for word, tag in tagged:
        if tag not in _VALID_TAGS:
            continue
        if word in _STOPWORDS or not word.isalpha() or len(word) < 2:
            continue
        wn_pos = _penn_to_wordnet(tag)
        lemma = _LEMMATIZER.lemmatize(word, pos=wn_pos)
        if lemma not in _STOPWORDS:
            concepts.add(lemma)
    return concepts


def build_concept_universe(
    documents: list[Document], top_l: int = 20
) -> ConceptUniverse:
    sorted_docs = sorted(documents, key=lambda d: d.relevance_score, reverse=True)
    top_docs = sorted_docs[:top_l]

    # Build universe from top-L documents
    concept_weights: dict[str, float] = {}
    for doc in top_docs:
        concepts = extract_concepts_from_text(doc.text)
        for c in concepts:
            current_weight = concept_weights.get(c, 0.0)
            concept_weights[c] = max(current_weight, doc.relevance_score)

    all_concepts = set(concept_weights.keys())

    # Map ALL documents to concepts within the universe
    doc_concepts: dict[str, set[str]] = {}
    for doc in sorted_docs:
        extracted = extract_concepts_from_text(doc.text)
        doc_concepts[doc.doc_id] = extracted & all_concepts

    return ConceptUniverse(
        concepts=all_concepts,
        concept_weights=concept_weights,
        doc_concepts=doc_concepts,
    )
