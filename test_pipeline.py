"""
test_pipeline.py
----------------
Run this to verify the full analysis pipeline works correctly
WITHOUT launching the Streamlit UI.

Usage:
    python test_pipeline.py
"""

import sys
import os

# Make sure we can import our utils from anywhere
sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessor       import preprocess_text, get_word_count, get_unique_word_ratio
from utils.sentiment_analyzer  import analyze_sentiment
from utils.confidence_scorer   import compute_confidence_score
from utils.communication_scorer import compute_communication_score
from utils.feedback_generator  import generate_feedback

# ── Test cases ────────────────────────────────────────────────────────────────
TEST_ANSWERS = [
    {
        "label": "Strong Answer",
        "question": "Tell me about yourself.",
        "answer": (
            "I have five years of experience in full-stack development, where I have led "
            "cross-functional teams to deliver high-impact products. At my last company, I "
            "built a real-time analytics dashboard that reduced reporting time by 60%. I am "
            "passionate about clean architecture, mentoring junior developers, and continuously "
            "improving systems. I am confident that my skills and enthusiasm make me a strong "
            "fit for this role."
        )
    },
    {
        "label": "Weak Answer (fillers + hedging)",
        "question": "What are your strengths?",
        "answer": (
            "Um, I think I'm like a pretty good team player, you know? I basically work hard "
            "and maybe I could be a leader someday, sort of. I'm not really sure what my "
            "biggest strength is, I guess maybe just trying my best."
        )
    },
    {
        "label": "Negative Tone",
        "question": "Describe a challenge you faced.",
        "answer": (
            "I failed a big project because my manager was terrible and the team never listened. "
            "It was awful and I struggled the whole time. I hated working there and eventually "
            "quit because it was impossible to succeed."
        )
    },
]

# ── Run pipeline ──────────────────────────────────────────────────────────────
SEP = "=" * 65

for test in TEST_ANSWERS:
    text = test["answer"]
    print(f"\n{SEP}")
    print(f"SCENARIO  : {test['label']}")
    print(f"QUESTION  : {test['question']}")
    print(f"ANSWER    : {text[:80]}...")
    print(SEP)

    # 1. Preprocess
    tokens = preprocess_text(text)
    print(f"\n[1] PREPROCESSED TOKENS ({len(tokens)} tokens)")
    print(f"    {tokens[:12]} {'...' if len(tokens) > 12 else ''}")

    # 2. Sentiment
    sent = analyze_sentiment(text)
    print(f"\n[2] SENTIMENT ANALYSIS")
    print(f"    Label    : {sent['label']}")
    print(f"    Score    : {sent['score']}/100")
    print(f"    Polarity : {sent['polarity']}")
    print(f"    Pos/Neg  : +{sent['pos_hits']} / -{sent['neg_hits']} keywords")

    # 3. Confidence
    conf = compute_confidence_score(text)
    print(f"\n[3] CONFIDENCE SCORING")
    print(f"    Score     : {conf['score']}/100")
    print(f"    Level     : {conf['level']}")
    print(f"    Fillers   : {conf['filler_count']}")
    print(f"    Hedges    : {conf['hedge_count']}")
    print(f"    Assertive : {conf['assertive_count']}")
    print(f"    Passive   : {conf['passive_count']}")
    print(f"    Words     : {conf['word_count']}")

    # 4. Communication
    comm = compute_communication_score(text)
    print(f"\n[4] COMMUNICATION QUALITY")
    print(f"    Score       : {comm['score']}/100")
    print(f"    Rating      : {comm['rating']}")
    print(f"    Grammar     : {comm['grammar']}")
    print(f"    Vocabulary  : {comm['vocabulary']}")
    print(f"    Structure   : {comm['structure']}")
    print(f"    Readability : {comm['readability']:.1f}")

    # 5. Final Score
    final = round(
        sent["score"]  * 0.30 +
        conf["score"]  * 0.40 +
        comm["score"]  * 0.30
    )
    print(f"\n[5] FINAL SCORE  →  {final}/100")

    # 6. Feedback
    fb = generate_feedback(sent, conf, comm, final)
    print(f"\n[6] FEEDBACK")
    print("    ✅ Positives:")
    for p in fb["positives"]:
        print(f"       • {p}")
    print("    → Suggestions:")
    for s in fb["suggestions"]:
        print(f"       • {s}")

print(f"\n{SEP}")
print("✅ All tests passed! Launch the UI with: streamlit run app.py")
print(SEP)
