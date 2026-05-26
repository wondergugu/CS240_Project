"""Download required NLTK data. Run once before using the project."""

import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("stopwords")
print("Done.")
