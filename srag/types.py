"""Core data types for budgeted document selection."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    relevance_score: float
    token_count: int

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if not isfinite(self.relevance_score) or self.relevance_score < 0:
            raise ValueError("relevance_score must be finite and nonnegative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    query_id: str = ""


@dataclass(slots=True)
class ConceptUniverse:
    concepts: frozenset[str] = field(default_factory=frozenset)
    concept_weights: dict[str, float] = field(default_factory=dict)
    doc_concepts: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.concepts = frozenset(self.concepts)
        self.doc_concepts = {
            doc_id: frozenset(concepts)
            for doc_id, concepts in self.doc_concepts.items()
        }

        weighted = set(self.concept_weights)
        if weighted != self.concepts:
            raise ValueError(
                "concept_weights must define exactly the concept universe"
            )

        for concept, weight in self.concept_weights.items():
            if not isfinite(weight) or weight < 0:
                raise ValueError(
                    f"concept weight for {concept!r} must be nonnegative"
                )

        for doc_id, concepts in self.doc_concepts.items():
            if not concepts.issubset(self.concepts):
                raise ValueError(
                    f"document {doc_id!r} covers concepts outside universe"
                )
