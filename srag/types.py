from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    relevance_score: float
    token_count: int

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        return isinstance(other, Document) and self.doc_id == other.doc_id


@dataclass
class Query:
    text: str
    query_id: str = ""


@dataclass
class ConceptUniverse:
    concepts: set = field(default_factory=set)
    concept_weights: dict = field(default_factory=dict)
    doc_concepts: dict = field(default_factory=dict)
