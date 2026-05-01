# 🎯 AI Interview Analyzer

> An end-to-end Machine Learning project that evaluates interview answers across
> three AI-powered dimensions: **Sentiment**, **Confidence**, and **Communication Quality**.

---

## 📁 Folder Structure

```
interview_analyzer/
│
├── app.py                    ← Main Streamlit UI application
├── requirements.txt          ← Python dependencies
├── test_pipeline.py          ← Full pipeline test (run first)
├── demo_output.py            ← No-dependency demo (offline check)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py       ← Text cleaning, tokenization, stopword removal
│   ├── sentiment_analyzer.py ← TextBlob + keyword sentiment scoring
│   ├── confidence_scorer.py  ← Filler words, hedges, assertive language
│   ├── communication_scorer.py ← Grammar, vocabulary, structure, readability
│   └── feedback_generator.py ← Personalised feedback & suggestions
│
└── data/
    └── sample_answers.csv    ← 10 labelled sample interview answers
```

---

## 🚀 Quick Start

### 1. Clone / download the project
```bash
git clone https://github.com/yourname/interview-analyzer.git
cd interview_analyzer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### 5. Verify pipeline
```bash
python test_pipeline.py
```

### 6. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧠 How It Works — Module Breakdown

### Module 1: `utils/preprocessor.py` — Text Preprocessing

```
Raw Text → Lowercase → Remove URLs/punctuation → Tokenize → Remove stopwords → Clean tokens
```

| Function | Purpose |
|---|---|
| `preprocess_text(text)` | Full pipeline → list of clean tokens |
| `get_sentences(text)` | Split into sentences |
| `get_unique_word_ratio(text)` | Type-Token Ratio (vocab richness) |

---

### Module 2: `utils/sentiment_analyzer.py` — Sentiment Analysis (30% weight)

Uses **TextBlob** polarity score + **domain-specific keyword boosting**:

- Detects positive keywords: `achieved`, `led`, `passionate`, `improved`…
- Detects negative keywords: `failed`, `struggled`, `hate`, `quit`…
- Penalises hedging: `maybe`, `I think`, `not sure`…

**Score mapping:**
```
Positive ≥ 65  →  Label: "Positive 😊"
40–64          →  Label: "Neutral 😐"
< 40           →  Label: "Negative 😟"
```

---

### Module 3: `utils/confidence_scorer.py` — Confidence Analysis (40% weight)

Scores confidence through six signals:

| Signal | Effect |
|---|---|
| Filler words (um, uh, like) | **Penalty** −5 pts each |
| Hedge phrases (I think, maybe) | **Penalty** −3 pts each |
| Assertive phrases (I led, I built) | **Reward** +4 pts each |
| Answer length (ideal: 80–200 words) | Base score |
| Passive voice (was done by me) | **Penalty** −4 pts each |
| Sentence length variety | Small reward |

---

### Module 4: `utils/communication_scorer.py` — Communication Quality (30% weight)

Five sub-scores:

| Sub-score | Weight | What it Measures |
|---|---|---|
| Grammar | 30% | TextBlob spelling correction diff |
| Vocabulary | 25% | Type-Token Ratio (unique words / total words) |
| Structure | 25% | Transition word count (firstly, however, in conclusion…) |
| Readability | 10% | Flesch Reading Ease formula |
| Sentence Variety | 10% | Standard deviation of sentence lengths |

---

### Module 5: `utils/feedback_generator.py` — Feedback Engine

Generates **personalised** feedback (not hardcoded) based on actual scores:

```python
# Example: Only shows filler word tip if count > 0
if filler_count >= 3:
    suggestions.append(f"You used {filler_count} filler words...")
```

Returns two lists:
- ✅ **Positives** — what the candidate did well
- → **Suggestions** — specific, actionable improvements

---

## 📊 Scoring Formula

```
Final Score = (Sentiment × 0.30) + (Confidence × 0.40) + (Communication × 0.30)
```

| Grade | Score Range |
|---|---|
| 🌟 Excellent | 75–100 |
| 👍 Good | 55–74 |
| 📈 Fair | 35–54 |
| 💪 Needs Work | 0–34 |

---

## 📋 Sample Outputs

### Input 1: Strong Answer
```
"I have five years of experience in software development, where I led a team 
to build a payment platform that reduced errors by 40%. I am passionate about 
clean architecture and I am confident my background makes me a strong fit."
```
```
Sentiment     : Positive 😊  →  95/100
Confidence    : High 💪       →  87/100  
Communication : Good 👍       →  70/100
─────────────────────────────────────
FINAL SCORE   : EXCELLENT 🌟  →  84/100
```

### Input 2: Weak Answer (fillers)
```
"Um, I think I'm like a pretty good worker, you know? I basically try my best 
and maybe I could be a leader someday, sort of."
```
```
Sentiment     : Neutral 😐    →  50/100
Confidence    : Low 😬        →  13/100
Communication : Average 📊    →  55/100
─────────────────────────────────────
FINAL SCORE   : FAIR 📈       →  34/100
```

---

## 🗃️ Dataset

`data/sample_answers.csv` contains **10 labelled interview answers** with:
- Answer text
- Question
- Label (positive_confident, low_confidence_filler, etc.)
- Expected score range

To create your own larger dataset:
1. Collect answers from friends, online forums, or YouTube transcripts
2. Label them manually (1–5 quality scale)
3. Use them to fine-tune a sklearn Logistic Regression classifier

---

## 🔧 Tech Stack

| Library | Used For |
|---|---|
| **Streamlit** | Web UI |
| **TextBlob** | Sentiment polarity, grammar approximation |
| **NLTK** | Tokenization, stopword removal |
| **Plotly** | Radar chart, visualisations |
| **scikit-learn** | TF-IDF (extendable for ML classification) |

---

## 🚀 Bonus Features (Implemented)

- ✅ **10 interview question bank** with selector
- ✅ **Radar chart** showing three-axis score visualisation  
- ✅ **Real-time word counter** and answer statistics
- ✅ **Tips expander** with STAR method guidance
- ✅ **Four answer statistics**: words, sentences, unique words, fillers

---

## 💡 Suggestions for Future Improvement

1. **Voice Input**: Use `speech_recognition` library or browser Web Speech API
2. **ML Classifier**: Train a Logistic Regression on 500+ labelled answers using TF-IDF features
3. **Industry-Specific Models**: Separate models for tech, management, sales interviews
4. **Score History**: Save past analyses to SQLite and show progress over time
5. **PDF Export**: Generate a full interview performance report
6. **Multi-Language**: Extend to Hindi/regional language interviews
7. **Video Analysis**: Add facial expression / eye contact analysis via OpenCV
8. **LLM Integration**: Use Claude/GPT API to generate richer feedback

---

## 👨‍💻 Author

Built as a college ML Portfolio Project.  
Tech: Python · NLTK · TextBlob · Streamlit · Plotly
