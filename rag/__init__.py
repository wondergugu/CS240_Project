"""End-to-end RAG evaluation layer for the S-RAG project.

This package wraps the document-selection core in `srag/` with a retrieval
corpus, a local generator, answer-quality metrics, and an open LLM judge so
that selection strategies (S-RAG / MMR / Top-k / Greedy) can be compared on
final answer quality, following the paper's experimental protocol.

Compliance: every model used here is open-weight and runs locally. The system
never calls a commercial LLM API.
"""
