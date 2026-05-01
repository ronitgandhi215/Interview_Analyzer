<div align="center">

# Interview Analyzer

**An AI-powered system that evaluates interview answers using NLP and Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-4CAF50?style=flat-square)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br>

_Type or speak your interview answer — get scored on sentiment, confidence, and communication quality in seconds._

<br>

</div>

---

## What It Does

Interview Analyzer takes a candidate's answer to an interview question — typed or spoken — and evaluates it across two AI layers:

**NLP Layer** — rule-based linguistic analysis using TextBlob and NLTK

- Sentiment (positive, neutral, or negative tone)
- Confidence (filler words, hedging, assertive language, passive voice)
- Communication quality (grammar, vocabulary richness, structure, readability)

**ML Layer** — trained Logistic Regression classifier using TF-IDF features

- Predicts answer quality class: Poor / Average / Good / Excellent
- Returns probability distribution across all four classes
- 5-fold cross-validated on 120 labelled interview answers

Both scores are combined into a single weighted final score out of 100, along with personalised feedback drawn from a pool of 200+ unique suggestions.

---

## Screenshots

<div align="center">

|                                       Input Screen                                        |                                      Analysis Results                                       |
| :---------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Input](https://github.com/user-attachments/assets/7a58be84-31f2-4f98-8512-c7b2686c5c88) | ![Results](https://github.com/user-attachments/assets/38b516cc-2288-4148-b3f7-b96da3a64aaa) |

</div>

---

## Architecture

```
User Input (Text or Voice)
         │
         ▼
┌─────────────────────┐
│   Preprocessor      │  Lowercase → Tokenize → Remove stopwords
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌───────┐   ┌─────────────────────────┐
│  NLP  │   │    ML Classification    │
│ Layer │   │  TF-IDF + Logistic      │
│       │   │  Regression (OvR)       │
│ • Sentiment (30%)  │  • Poor         │
│ • Confidence (40%) │  • Average      │
│ • Communication    │  • Good         │
│         (30%)      │  • Excellent    │
└───┬───┘   └──────────┬──────────────┘
    │                  │
    └────────┬─────────┘
             ▼
   Final Score (NLP 55% + ML 45%)
             │
             ▼
   Feedback + Suggestions + Charts
```

---

## MLA Component (Academic)

This project was built as an MLA (Machine Learning Applications) mini project. The ML component includes:

| Concept            | Implementation                                            |
| ------------------ | --------------------------------------------------------- |
| Feature Extraction | TF-IDF Vectorizer — unigrams + bigrams, top 2000 features |
| Classification     | Logistic Regression — One-vs-Rest multi-class strategy    |
| Regularization     | C=1.0 controls bias-variance trade-off                    |
| Evaluation         | Accuracy, Precision, Recall, F1-score, Confusion Matrix   |
| Validation         | 5-fold Stratified K-Fold Cross Validation                 |
| Probability        | `predict_proba()` returns confidence per class            |
| Ensemble           | NLP rule-based + ML learned — combined final score        |

Running `python model/train_model.py` prints the full classification report to terminal — suitable for academic submission.

---

## Project Structure

```
interview_analyzer/
│
├── app.py                        ← Streamlit UI (main entry point)
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── train_model.py            ← Train TF-IDF + Logistic Regression
│   ├── predictor.py              ← Load model, predict class + probabilities
│   ├── logistic_regression_model.pkl   ← Saved trained pipeline
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_evaluation_report.txt
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py           ← Tokenization, stopword removal
│   ├── sentiment_analyzer.py     ← TextBlob + keyword scoring
│   ├── confidence_scorer.py      ← Filler words, hedges, assertive language
│   ├── communication_scorer.py   ← Grammar, vocabulary, structure, readability
│   ├── feedback_generator.py     ← 200+ suggestion pool, context-sensitive
│   └── voice_input.py            ← SpeechRecognition + pyaudio
│
└── data/
    ├── sample_answers.csv         ← Example answers for testing
    └── training_data.csv          ← 120 labelled answers (4 classes)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- For voice input on Mac: `brew install portaudio`

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/interview-analyzer.git
cd interview-analyzer
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download NLTK data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

**5. Train the ML model**

```bash
python model/train_model.py
```

You will see the full classification report printed in the terminal.

**6. Launch the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Voice Input Setup

Voice input uses SpeechRecognition with Google Speech API (free, no key needed).

**Mac**

```bash
brew install portaudio
pip install SpeechRecognition pyaudio
```

**Windows**

```bash
pip install SpeechRecognition pyaudio
```

**Linux**

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install SpeechRecognition pyaudio
```

---

## Scoring System

```
Final Score = NLP Score (55%) + ML Score (45%)

NLP Score   = Sentiment (30%) + Confidence (40%) + Communication (30%)

ML Score    = Weighted probability across Poor/Average/Good/Excellent classes
```

| Grade      | Score    |
| ---------- | -------- |
| Excellent  | 75 – 100 |
| Good       | 55 – 74  |
| Fair       | 35 – 54  |
| Needs Work | 0 – 34   |

---

## Sample Output

**Strong Answer**

```
"I led a team of five engineers to build a payment platform that reduced
transaction errors by 40 percent. I am passionate about clean architecture
and I am confident my background makes me a strong fit for this role."

Sentiment     → Positive    95 / 100
Confidence    → High        87 / 100
Communication → Good        74 / 100
ML Prediction → Excellent   (confidence: 82%)
─────────────────────────────────────────
Final Score   → 84 / 100   EXCELLENT
```

**Weak Answer**

```
"Um, I think I'm like a pretty good worker, you know? I basically try
my best and maybe I could be a leader someday, sort of."

Sentiment     → Neutral     50 / 100
Confidence    → Low         13 / 100
Communication → Average     55 / 100
ML Prediction → Poor        (confidence: 71%)
─────────────────────────────────────────
Final Score   → 34 / 100   NEEDS WORK
```

---

## Adding More Training Data

Open `data/training_data.csv` and add rows following this format:

```csv
answer,label
"Your interview answer here",Excellent
"Another answer",Good
"A mediocre answer",Average
"A weak answer with um and uh",Poor
```

Labels must be exactly: `Poor`, `Average`, `Good`, `Excellent`

After adding data, retrain:

```bash
python model/train_model.py
```

More data directly improves ML accuracy. Current dataset: 120 samples (~85% CV accuracy). Recommended minimum: 200 samples per class.

---

## Requirements

```
streamlit>=1.32.0
textblob>=0.18.0
nltk>=3.8.1
plotly>=5.20.0
scikit-learn>=1.4.0
pandas>=2.0.0
SpeechRecognition>=3.10.0
pyaudio>=0.2.13
```

---

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Ideas for contribution:

- Add more training data to improve ML accuracy
- Add support for more languages
- Implement BERT-based scoring for richer NLP
- Add PDF export for analysis reports
- Build a progress tracker across multiple sessions

---

## Future Improvements

- Fine-tuned BERT model for deeper semantic understanding
- Video analysis — facial expression and eye contact scoring
- Multi-language support beyond English and Hindi
- Session history — track improvement over time
- PDF report generation for interview preparation portfolios
- Domain-specific models — tech, sales, management interviews

---

## Tech Stack

<div align="center">

| Layer         | Technology                                 |
| ------------- | ------------------------------------------ |
| UI            | Streamlit                                  |
| NLP           | TextBlob, NLTK                             |
| ML            | scikit-learn (TF-IDF, Logistic Regression) |
| Voice         | SpeechRecognition, pyaudio                 |
| Visualisation | Plotly                                     |
| Language      | Python 3.9+                                |

</div>

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis
- [NLTK](https://www.nltk.org/) for text preprocessing
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for interactive charts

---

<div align="center">

Built as an MLA Mini Project &nbsp;·&nbsp; Python · NLTK · TextBlob · scikit-learn · Streamlit

⭐ Star this repo if you found it useful

</div>
