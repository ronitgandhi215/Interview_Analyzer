<div align="center">

# Interview Analyzer

**An AI-powered tool that evaluates interview answers (typed or spoken) using NLP heuristics and an ML classifier.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br>

Type or speak your interview answer — get a combined score for sentiment, confidence, and communication quality, plus ML-based quality prediction and actionable feedback.

</div>

---

## Highlights / Features

- Input modes: `Type` or `Speak` (microphone).
- Voice recording features:
  - Simple visible countdown timer while recording.
  - `Stop` button to end early and save captured audio.
  - `Transcribe` button after recording to run offline Whisper transcription (faster-whisper).
  - Edit transcript inline before analysis.
- Improved transcription settings (higher beam size, VAD disabled by default) for better accuracy on short answers.
- NLP scoring: Sentiment, Confidence, Communication (weighted to form an NLP score).
- ML scoring: TF-IDF + Logistic Regression predicts quality class (Poor / Average / Good / Excellent) and returns class probabilities.
- Combined final score and contextual feedback suggestions.
- Mobile-friendly responsive layout and polished UI styling.

Note: The on-screen "Save to history" button was removed in this release; helper functions remain in the codebase if you want to re-enable persistent history later.

---

## Quick Start

Follow these steps to run locally and test the app.

### Prerequisites

- Python 3.9 or later
- `pip`
- (Optional, for voice) system PortAudio headers:
  - macOS: `brew install portaudio`
  - Ubuntu: `sudo apt-get install libportaudio2 portaudio19-dev`

### Install & run

```bash
git clone https://github.com/yourusername/interview-analyzer.git
cd interview-analyzer
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

If you plan to use offline voice transcription, also install the voice dependencies:

```bash
pip install sounddevice soundfile faster-whisper
```

Finally, run the app with:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

---

## Voice Input Details

- Recording flow:
  1. Choose `Speak` mode, select language and duration.
  2. Click `Record` — a countdown shows remaining seconds.
  3. Click `Stop` any time to finish early or wait until the countdown ends.
  4. Click `Transcribe` to convert the saved audio into text.
  5. Use `Edit transcript` to refine text before analysis.

- Implementation notes:
  - Recording uses `sounddevice` and writes WAV files via `soundfile`.
  - Transcription runs locally with `faster-whisper` (Whisper model). Default settings favour accuracy (larger beam size) and avoid aggressive VAD filtering that may drop words in short responses.
  - If your environment cannot access a microphone or dependencies are missing, the UI will display a clear warning and instructions.

---

## Analysis & Scoring

- NLP Score (55% weight in final score):
  - Sentiment (30%) — tone and valence
  - Confidence (40%) — filler words, hedging, assertiveness
  - Communication (30%) — grammar, vocabulary, readability

- ML Score (45% weight):
  - TF-IDF + Logistic Regression predicts class probabilities and a predicted label.

- Final score = weighted combination of NLP and ML scores; app maps to grade buckets (Excellent / Good / Fair / Needs Work).

---

## Model Training

To retrain or improve the ML model:

```bash
python model/train_model.py
```

Training reads `data/training_data.csv` (format: `answer,label`) and writes model artifacts to `model/`.

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'streamlit'` — install dependencies and activate the correct virtual environment.
- Microphone not detected — ensure OS permissions allow microphone access and that `sounddevice` lists input devices.
- If transcription is noisy:
  - Try switching to a larger Whisper model in `utils/voice_input.py` (e.g., `base` or `small`) if you have CPU/RAM budget.
  - Reduce background noise and speak clearly close to the mic.

---

## Files of interest

- `app.py` — Streamlit UI and control flow (record → transcribe → edit → analyze)
- `utils/voice_input.py` — recording + transcription helpers (supports chunked recording, stop events, and status updates)
- `model/train_model.py` — training pipeline for TF-IDF + Logistic Regression
- `model/predictor.py` — model loading and `predict_quality()` API used by the app

---

## Requirements (important)

Minimum packages for core app (non-voice):

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
plotly>=5.20.0
nltk>=3.8.1
textblob>=0.18.0
```

Optional voice/transcription packages:

```
sounddevice
soundfile
faster-whisper
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit and push, then open a PR

Contributions: more labeled training data, multi-language support, better NLP features, or a progress tracker.

---

## License

This project is licensed under MIT. See [LICENSE](LICENSE).

---

Built as an MLA Mini Project · Streamlit · scikit-learn · faster-whisper

⭐ Star the repo if it helped you
