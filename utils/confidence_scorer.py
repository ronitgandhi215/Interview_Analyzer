"""
utils/confidence_scorer.py
---------------------------
Measures how confident the speaker sounds by analysing:

  1. Filler word count   (um, uh, like, you know…)  → penalises score
  2. Hedge word count    (maybe, I think, kind of…)  → mild penalty
  3. Assertive language  (I will, I have, I did…)    → rewards score
  4. Sentence length     (too short = lack of depth) → rewards score
  5. Answer length       (very short = not enough)   → penalises score
  6. Passive voice usage (detected via auxiliaries)  → mild penalty

Final weight in overall score: 40%
"""

import re
from utils.preprocessor import get_sentences, get_word_count


# ── Word lists ─────────────────────────────────────────────────────────────────
FILLER_WORDS = [
    "um", "uh", "umm", "uhh", "err", "like", "you know", "basically",
    "literally", "actually", "right", "so yeah", "i mean", "kind of",
    "sort of", "well", "okay so", "maybe"
]

HEDGE_PHRASES = [
    "i think", "i believe", "maybe", "perhaps", "possibly", "not sure",
    "i guess", "probably", "might", "could be", "i feel like"
]

ASSERTIVE_PHRASES = [
    "i have", "i did", "i will", "i led", "i built", "i created",
    "i achieved", "i managed", "i delivered", "i improved",
    "i am confident", "i am experienced", "i successfully",
    "i consistently", "my experience", "i specialize"
]

PASSIVE_PATTERNS = [
    r"\bwas\s+\w+ed\b",
    r"\bwere\s+\w+ed\b",
    r"\bbeen\s+\w+ed\b",
    r"\bis\s+being\b"
]


def _count_list_hits(text: str, word_list: list) -> int:
    """Count how many phrases from word_list appear in text (case-insensitive)."""
    lower = text.lower()
    return sum(1 for w in word_list if w in lower)


def _count_passive_hits(text: str) -> int:
    lower = text.lower()
    return sum(len(re.findall(p, lower)) for p in PASSIVE_PATTERNS)


def compute_confidence_score(text: str) -> dict:
    """
    Compute a confidence score (0–100) for the given interview answer.

    Returns a dict:
        {
            "score":        int (0–100),
            "level":        str ("High" | "Medium" | "Low"),
            "filler_count": int,
            "hedge_count":  int,
            "assertive_count": int,
            "breakdown":    dict  (for detailed UI display)
        }
    """
    word_count   = get_word_count(text)
    sentences    = get_sentences(text)
    num_sentences = max(len(sentences), 1)
    avg_sent_len  = word_count / num_sentences

    filler_count    = _count_list_hits(text, FILLER_WORDS)
    hedge_count     = _count_list_hits(text, HEDGE_PHRASES)
    assertive_count = _count_list_hits(text, ASSERTIVE_PHRASES)
    passive_count   = _count_passive_hits(text)

    # ── Base score from length (ideal: 80–200 words) ──────────────────────────
    if word_count < 20:
        length_score = 20
    elif word_count < 50:
        length_score = 45
    elif word_count < 80:
        length_score = 62
    elif word_count <= 200:
        length_score = 78
    elif word_count <= 300:
        length_score = 70  # too long = losing focus
    else:
        length_score = 55

    # ── Penalties ─────────────────────────────────────────────────────────────
    filler_penalty   = min(filler_count  * 5,  30)
    hedge_penalty    = min(hedge_count   * 3,  15)
    passive_penalty  = min(passive_count * 4,  12)

    # ── Rewards ───────────────────────────────────────────────────────────────
    assertive_bonus = min(assertive_count * 4, 20)
    sentence_bonus  = min((avg_sent_len - 8) * 0.5, 8) if avg_sent_len > 8 else 0

    # ── Final score ───────────────────────────────────────────────────────────
    raw = length_score + assertive_bonus + sentence_bonus - filler_penalty - hedge_penalty - passive_penalty
    score = int(max(0, min(100, raw)))

    # ── Level label ──────────────────────────────────────────────────────────
    if score >= 70:
        level   = "High Confidence 💪"
        detail  = "You speak with authority. Clear, assertive language dominates."
    elif score >= 45:
        level   = "Medium Confidence 🤔"
        detail  = "Some confidence shown, but hedging or filler words weaken impact."
    else:
        level   = "Low Confidence 😬"
        detail  = "Too many filler words or hedges reduce perceived confidence."

    return {
        "score":          score,
        "level":          level,
        "detail":         detail,
        "filler_count":   filler_count,
        "hedge_count":    hedge_count,
        "assertive_count": assertive_count,
        "passive_count":  passive_count,
        "word_count":     word_count,
        "breakdown": {
            "length_score":     length_score,
            "assertive_bonus":  assertive_bonus,
            "filler_penalty":   -filler_penalty,
            "hedge_penalty":    -hedge_penalty,
            "passive_penalty":  -passive_penalty,
        }
    }
