"""
demo_output.py
--------------
Self-contained demo that simulates the AI Interview Analyzer output
WITHOUT requiring textblob/nltk — useful to verify logic & show sample output.

On your machine, run `test_pipeline.py` after `pip install -r requirements.txt`.
"""

import re
import math

# ── Minimal inline implementations (no external deps) ─────────────────────────

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","is","was",
    "it","this","that","with","as","by","from","are","were","be","been","have",
    "has","had","do","does","did","will","would","could","should","may","might",
    "i","my","me","we","our","you","your","they","their","he","she","his","her"
}

FILLER_WORDS = ["um","uh","umm","uhh","like","you know","basically","literally","actually","right","so yeah","i mean","kind of","sort of"]
HEDGE_PHRASES = ["i think","i believe","maybe","perhaps","possibly","not sure","i guess","probably","might","could be","i feel like"]
ASSERTIVE_PHRASES = ["i have","i did","i will","i led","i built","i created","i achieved","i managed","i delivered","i improved","i am confident","my experience","i specialize","i successfully"]
POSITIVE_KEYWORDS = ["achieved","accomplished","improved","led","developed","created","built","launched","managed","delivered","exceeded","increased","motivated","passionate","excited","proud","successful","excellent","dedicated","innovative","collaborative","solution","growth","opportunity","learn","contribute","committed"]
NEGATIVE_KEYWORDS = ["failed","struggled","quit","fired","hate","terrible","awful","worst","dislike","bored","unmotivated","lazy","confused","lost","hopeless","impossible"]
TRANSITION_WORDS = ["firstly","secondly","finally","therefore","however","moreover","additionally","consequently","in conclusion","as a result","for example","for instance","in addition","furthermore","on the other hand","to summarize","specifically","notably","importantly","to begin with","next","then","subsequently","ultimately"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]"," ",text)
    return [t for t in text.split() if t.isalpha() and t not in STOP_WORDS and len(t)>1]

def get_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]+",text) if s.strip()]

def analyze_sentiment(text):
    lower = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)
    # Simple rule-based polarity
    polarity = (pos - neg) / max(pos + neg, 1)
    score = int(max(0, min(100, 50 + polarity * 45)))
    if score >= 65:   label = "Positive 😊"
    elif score >= 40: label = "Neutral 😐"
    else:             label = "Negative 😟"
    return {"score": score, "label": label, "pos_hits": pos, "neg_hits": neg, "polarity": round(polarity,2)}

def compute_confidence_score(text):
    lower = text.lower()
    word_count = len(text.split())
    sentences = get_sentences(text)
    fillers = sum(1 for w in FILLER_WORDS if w in lower)
    hedges = sum(1 for h in HEDGE_PHRASES if h in lower)
    assertive = sum(1 for a in ASSERTIVE_PHRASES if a in lower)
    
    if word_count < 20:  length_s = 25
    elif word_count < 50: length_s = 50
    elif word_count <= 200: length_s = 75
    else: length_s = 60
    
    raw = length_s + assertive*4 - fillers*5 - hedges*3
    score = int(max(0,min(100,raw)))
    if score>=70: level="High Confidence 💪"
    elif score>=45: level="Medium Confidence 🤔"
    else: level="Low Confidence 😬"
    return {"score":score,"level":level,"filler_count":fillers,"hedge_count":hedges,"assertive_count":assertive,"passive_count":0,"word_count":word_count}

def compute_communication_score(text):
    lower = text.lower()
    words = text.split()
    unique_ratio = len(set(lower.split())) / max(len(lower.split()),1)
    transitions = sum(1 for t in TRANSITION_WORDS if t in lower)
    
    vocab_s = min(unique_ratio * 130, 100)
    struct_s = min(40 + transitions*15, 100)
    grammar_s = 72  # approximation without spellcheck lib
    
    score = int(vocab_s*0.35 + struct_s*0.35 + grammar_s*0.30)
    score = max(0, min(100, score))
    
    if score>=80: rating="Excellent Communicator 🌟"
    elif score>=65: rating="Good Communicator 👍"
    elif score>=45: rating="Average Communicator 📊"
    else: rating="Needs Improvement 📝"
    return {"score":score,"rating":rating,"grammar":grammar_s,"vocabulary":round(vocab_s,1),"structure":round(struct_s,1),"readability":65.0}

def generate_feedback(sent, conf, comm, final):
    positives, suggestions = [], []
    if sent["score"]>=65: positives.append("Positive and enthusiastic tone — great first impression.")
    if conf["assertive_count"]>=2: positives.append("Strong assertive language signals ownership and confidence.")
    if conf["filler_count"]==0: positives.append("Zero filler words — your answer sounds polished.")
    if conf["word_count"]>=80: positives.append(f"Good length ({conf['word_count']} words) — enough detail to be convincing.")
    if comm["vocabulary"]>=65: positives.append("Varied vocabulary demonstrates strong language skills.")
    if not positives: positives.append("You provided an answer — that's the first step to improvement.")
    
    if conf["filler_count"]>=3: suggestions.append(f"Reduce {conf['filler_count']} filler words. Pause silently instead of saying 'um' or 'like'.")
    if conf["hedge_count"]>=3: suggestions.append("Replace hedging ('I think', 'maybe') with direct statements.")
    if conf["word_count"]<50: suggestions.append("Answer is too short. Use the STAR method to add detail and depth.")
    if sent["score"]<50: suggestions.append("Reframe negatives as learnings. Focus on growth, not difficulties.")
    if comm["structure"]<50: suggestions.append("Add transitions ('firstly', 'as a result', 'in conclusion') for better structure.")
    if 35<=final<=70: suggestions.append("Practice the STAR method: Situation → Task → Action → Result.")
    if not suggestions: suggestions.append("Record yourself and replay to spot any remaining weak points.")
    return {"positives":positives,"suggestions":suggestions}

# ── DEMO RUN ──────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "label": "✅ STRONG ANSWER",
        "answer": "I have five years of software development experience, where I led a team to build a payment platform that reduced errors by 40%. I am passionate about clean architecture and continuously improving systems. I have successfully delivered multiple high-impact projects under tight deadlines and I am confident this experience makes me an excellent fit."
    },
    {
        "label": "⚠️ WEAK ANSWER (fillers + hedging)",
        "answer": "Um, I think I'm like a pretty good worker, you know? I basically try my best and maybe I could be a leader someday, sort of. I'm not really sure what my biggest strength is, I guess."
    },
    {
        "label": "❌ NEGATIVE TONE",
        "answer": "I failed a big project because my manager was terrible. It was awful and I struggled the whole time. I hated working there and quit because it was impossible."
    }
]

BAR = "═" * 66

for tc in TEST_CASES:
    text = tc["answer"]
    print(f"\n{BAR}")
    print(f"  {tc['label']}")
    print(BAR)
    print(f"  Answer: \"{text[:70]}{'...' if len(text)>70 else ''}\"")
    print()

    sent = analyze_sentiment(text)
    conf = compute_confidence_score(text)
    comm = compute_communication_score(text)
    final = round(sent["score"]*0.30 + conf["score"]*0.40 + comm["score"]*0.30)
    fb = generate_feedback(sent, conf, comm, final)

    print(f"  ┌─ SENTIMENT    {sent['label']:<25}  Score: {sent['score']:>3}/100")
    print(f"  ├─ CONFIDENCE   {conf['level']:<25}  Score: {conf['score']:>3}/100")
    print(f"  ├─ COMMUNICATION {comm['rating']:<24}  Score: {comm['score']:>3}/100")

    grade = "EXCELLENT 🌟" if final>=75 else "GOOD 👍" if final>=55 else "FAIR 📈" if final>=35 else "NEEDS WORK 💪"
    print(f"  └─ FINAL SCORE  {grade:<25}  Score: {final:>3}/100")

    print(f"\n  ✅ Strengths:")
    for p in fb["positives"]: print(f"     • {p}")
    print(f"\n  → Improvements:")
    for s in fb["suggestions"]: print(f"     • {s}")

print(f"\n{BAR}")
print("  🚀 Run 'streamlit run app.py' after installing requirements!")
print(BAR)
