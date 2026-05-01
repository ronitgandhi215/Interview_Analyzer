"""
utils/communication_scorer.py
------------------------------
Measures communication quality via:

  1. Grammar / Spelling  – via TextBlob correction diff (approximation)
  2. Vocabulary Richness – Type-Token Ratio (unique words / total words)
  3. Sentence Variety    – avg sentence length + variance
  4. Readability         – Flesch Reading Ease (simplified)
  5. Structure           – transition / connective words detect

Final weight in overall score: 30%
"""

import re
import math
from textblob import TextBlob
from utils.preprocessor import get_sentences, get_word_count, get_unique_word_ratio


# ── Transition / connective words (show structured thinking) ──────────────────
TRANSITION_WORDS = [
    "firstly", "secondly", "finally", "therefore", "however", "moreover",
    "additionally", "consequently", "in conclusion", "as a result",
    "for example", "for instance", "in addition", "furthermore",
    "on the other hand", "in contrast", "to summarize", "in summary",
    "specifically", "notably", "importantly", "to begin with",
    "next", "then", "subsequently", "ultimately"
]


def _flesch_reading_ease(text: str) -> float:
    """
    Simplified Flesch Reading Ease score.
    Higher = easier to read (good for interviews: aim ~60–80).
    Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    """
    sentences = get_sentences(text)
    num_sents = max(len(sentences), 1)
    words = text.split()
    num_words = max(len(words), 1)

    # Rough syllable count: count vowel groups per word
    def count_syllables(word):
        word = word.lower()
        vowels = re.findall(r'[aeiouy]+', word)
        return max(len(vowels), 1)

    total_syllables = sum(count_syllables(w) for w in words)

    fre = 206.835 - 1.015 * (num_words / num_sents) - 84.6 * (total_syllables / num_words)
    return max(0, min(100, fre))


def _grammar_score(text: str) -> float:
    """
    Approximate grammar correctness using TextBlob spelling correction.
    Compares corrected vs original word counts. Not perfect, but sufficient
    for a student portfolio project.
    Returns a 0–100 score.
    """
    blob = TextBlob(text)
    original_words = text.lower().split()
    corrected_words = str(blob.correct()).lower().split()

    if not original_words:
        return 50.0

    # Count differing words
    min_len = min(len(original_words), len(corrected_words))
    differences = sum(1 for o, c in zip(original_words[:min_len], corrected_words[:min_len]) if o != c)
    error_rate = differences / len(original_words)
    return round(max(0, 100 - error_rate * 100), 1)


def _vocabulary_score(text: str) -> float:
    """
    Vocabulary richness score based on Type-Token Ratio (TTR).
    TTR = unique_words / total_words → higher = richer vocab.
    """
    ttr = get_unique_word_ratio(text)
    # Typical good interview TTR: 0.5–0.75
    return round(min(ttr * 130, 100), 1)


def _structure_score(text: str) -> float:
    """
    Detect use of transition/connective words → structured thinking.
    """
    lower = text.lower()
    hits = sum(1 for t in TRANSITION_WORDS if t in lower)
    # 3+ transitions = full marks; 0 = 40 base
    base = 40
    bonus = min(hits * 15, 60)
    return float(base + bonus)


def _sentence_variety_score(text: str) -> float:
    """
    Sentences of varied length signal good writing.
    Penalise: all very short OR one giant run-on sentence.
    """
    sentences = get_sentences(text)
    if len(sentences) <= 1:
        return 40.0

    lengths = [len(s.split()) for s in sentences]
    avg     = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    std_dev  = math.sqrt(variance)

    # Good variety: avg 10–20 words, std_dev > 3
    variety = min(std_dev * 4, 40)   # up to 40 points for variety
    avg_score = 60 if 8 <= avg <= 22 else 35  # 60 if good avg length

    return min(float(avg_score + variety), 100.0)


def compute_communication_score(text: str) -> dict:
    """
    Compute communication quality score (0–100).

    Returns a dict:
        {
            "score":      int,
            "rating":     str,
            "grammar":    float,
            "vocabulary": float,
            "structure":  float,
            "readability":float,
            "variety":    float,
            "detail":     str
        }
    """
    grammar_s     = _grammar_score(text)
    vocab_s       = _vocabulary_score(text)
    structure_s   = _structure_score(text)
    readability_s = _flesch_reading_ease(text)
    variety_s     = _sentence_variety_score(text)

    # Weighted average of sub-scores
    # Grammar 30% | Vocab 25% | Structure 25% | Readability 10% | Variety 10%
    raw = (
        grammar_s     * 0.30 +
        vocab_s       * 0.25 +
        structure_s   * 0.25 +
        readability_s * 0.10 +
        variety_s     * 0.10
    )
    score = int(max(0, min(100, raw)))

    # Rating label
    if score >= 80:
        rating = "Excellent Communicator 🌟"
        detail = "Clear, structured, and vocabulary-rich answer."
    elif score >= 65:
        rating = "Good Communicator 👍"
        detail = "Solid communication with minor room for polish."
    elif score >= 45:
        rating = "Average Communicator 📊"
        detail = "Some structure present but needs vocabulary and transitions."
    else:
        rating = "Needs Improvement 📝"
        detail = "Work on grammar, vocabulary, and structuring your thoughts."

    return {
        "score":       score,
        "rating":      rating,
        "detail":      detail,
        "grammar":     grammar_s,
        "vocabulary":  vocab_s,
        "structure":   structure_s,
        "readability": readability_s,
        "variety":     variety_s,
    }
