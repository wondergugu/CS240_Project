"""Dataset adapters: produce (question, reference answers) QA examples.

Questions and gold answers come from the QA data under data/. The retrieval
corpus is kept separate (see rag/corpus.py) so candidate pools never leak from
the answer data.
"""
