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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (runs once, cached after that)
for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Load English stopwords
STOP_WORDS = set(stopwords.words("english"))

# Keep a few meaningful words that NLTK marks as stopwords
KEEP_WORDS = {"not", "no", "never", "very", "too", "most", "more", "best", "great"}
STOP_WORDS -= KEEP_WORDS

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
    tokens = word_tokenize(text)

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
