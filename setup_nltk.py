"""Download and validate the NLTK data required by concept extraction."""

from zipfile import BadZipFile

import nltk


_RESOURCES = (
    ("punkt_tab", "tokenizers/punkt_tab/english/"),
    (
        "averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger_eng/",
    ),
    ("wordnet", "corpora/wordnet.zip"),
    ("stopwords", "corpora/stopwords/english"),
)


def _is_available(resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
    except (LookupError, BadZipFile):
        return False
    return True


def main() -> None:
    for package_id, resource_path in _RESOURCES:
        if _is_available(resource_path):
            print(f"{package_id}: ready")
            continue

        print(f"{package_id}: missing or corrupt; downloading")
        nltk.download(package_id, force=True, raise_on_error=True)
        if not _is_available(resource_path):
            raise RuntimeError(
                f"NLTK resource {package_id!r} remains unavailable or corrupt "
                "after download."
            )
        print(f"{package_id}: ready")

    print("All required NLTK resources are ready.")


if __name__ == "__main__":
    main()
