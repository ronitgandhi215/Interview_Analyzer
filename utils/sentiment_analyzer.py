"""
utils/sentiment_analyzer.py
----------------------------
Sentiment analysis using two complementary approaches:
  1. TextBlob polarity (fast, no training required)
  2. Keyword-based boosting for interview-specific language

Scoring:
  - Positive sentiment → high score (70–100)
  - Neutral            → medium score (40–69)
  - Negative           → low score (0–39)

Final weight in overall score: 30%
"""

from textblob import TextBlob
from utils.preprocessor import preprocess_text


# ── Interview-domain keyword lists ────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    "achieved", "accomplished", "improved", "led", "developed", "created",
    "built", "launched", "managed", "delivered", "exceeded", "increased",
    "motivated", "passionate", "excited", "proud", "successful", "excellent",
    "dedicated", "innovative", "collaborative", "solution", "growth",
    "opportunity", "learn", "contribute", "committed"
]

NEGATIVE_KEYWORDS = [
    "failed", "struggled", "quit", "fired", "hate", "terrible", "awful",
    "worst", "dislike", "bored", "uninspiring", "unmotivated", "lazy",
    "confused", "lost", "hopeless", "impossible", "never", "can't", "won't"
]

NEUTRAL_HEDGING = [
    "maybe", "perhaps", "possibly", "sort of", "kind of", "i think",
    "not sure", "i guess", "probably"
]


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of interview answer.

    Returns a dict:
        {
            "label":    "Positive" | "Neutral" | "Negative",
            "score":    int  (0–100),
            "polarity": float (-1 to +1),
            "detail":   str  (short human-readable explanation)
        }
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1.0 to +1.0

    # Count domain keywords
    lower_text = text.lower()
    tokens     = set(preprocess_text(text))

    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower_text)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower_text)
    hedge_hits = sum(1 for h in NEUTRAL_HEDGING if h in lower_text)

    # Adjusted polarity with keyword nudge
    keyword_nudge = (pos_hits - neg_hits) * 0.05
    adjusted_polarity = max(-1.0, min(1.0, polarity + keyword_nudge))

    # Convert polarity → 0–100 score
    #   polarity = -1  → score ≈ 0
    #   polarity =  0  → score ≈ 50
    #   polarity = +1  → score ≈ 100
    raw_score = (adjusted_polarity + 1) / 2 * 100

    # Penalise heavy hedging
    raw_score -= hedge_hits * 3

    score = int(max(0, min(100, raw_score)))

    # Label
    if score >= 65:
        label  = "Positive 😊"
        detail = f"Your answer reflects a positive and enthusiastic tone (+{pos_hits} strong keywords detected)."
    elif score >= 40:
        label  = "Neutral 😐"
        detail = "Your answer is balanced but lacks strong positive language to stand out."
    else:
        label  = "Negative 😟"
        detail = f"Your answer contains negative or deflating language ({neg_hits} negative keywords detected)."

    return {
        "label":    label,
        "score":    score,
        "polarity": round(adjusted_polarity, 3),
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "detail":   detail
    }
