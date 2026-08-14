"""
utils/preprocessor.py
---------------------
Text preprocessing pipeline:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stopword removal
  - (Optional) Stemming / Lemmatization
"""

import re

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

# Fallback stop words in case NLTK data is unavailable
FALLBACK_STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","is","was",
    "it","this","that","with","as","by","from","are","were","be","been","have",
    "has","had","do","does","did","will","would","could","should","may","might",
    "i","my","me","we","our","you","your","they","their","he","she","his","her"
}

KEEP_WORDS = {"not", "no", "never", "very", "too", "most", "more", "best", "great"}


def _load_stopwords() -> set[str]:
    if _NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = set()
    else:
        stop_words = set()

    if not stop_words:
        stop_words = FALLBACK_STOP_WORDS.copy()

    stop_words -= KEEP_WORDS
    return stop_words


def _tokenize(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[a-zA-Z]+", text)


STOP_WORDS = _load_stopwords()

# Keep a few meaningful words that NLTK marks as stopwords


def preprocess_text(text: str) -> list[str]:
    """
    Full preprocessing pipeline.

    Steps:
      1. Lowercase everything
      2. Remove URLs and special characters
      3. Tokenize into individual words
      4. Remove stopwords
      5. Keep only alphabetic tokens

    Returns:
        List of cleaned tokens (strings)
    """
    # Step 1 – Lowercase
    text = text.lower()

    # Step 2 – Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Step 3 – Remove punctuation / digits (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 – Tokenize
    tokens = _tokenize(text)

    # Step 5 – Filter: alphabetic + not a stopword + length > 1
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]

    return tokens


def get_sentences(text: str) -> list[str]:
    """Split raw text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_word_count(text: str) -> int:
    """Total word count (raw, no filtering)."""
    return len(text.split())


def get_unique_word_ratio(text: str) -> float:
    """
    Type–Token Ratio (TTR) = unique_words / total_words
    Higher → richer vocabulary.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)
