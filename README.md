<div align="center">

# Interview Analyzer

**AI-powered interview answer evaluator — type or speak, then get instant NLP + ML feedback.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br>

Type or speak your answer, edit the transcript, and get a combined score (Sentiment, Confidence, Communication) plus an ML-predicted quality class and actionable feedback.

</div>

---

## What's new in this release

- Browser-based recording using Streamlit's `st.audio_input` (no server audio hardware required).
- "Record again" button that resets the recorder so you can re-record from scratch.
- Automatic population of the main text input after transcription (so you can immediately edit and analyze).
- Whisper model is cached to avoid reloads and reduce out-of-memory issues in constrained environments.
- Simplified UI: the app shows only the recorder, `Transcribe`, and `Record again` controls for voice flows.

---

## Highlights / Features

- Input modes: `Type` or `Speak`.
- Browser recorder: `st.audio_input` (works in modern browsers; no server soundcard needed).
- Record again: clears previous recording and gives a fresh recorder instance.
- Transcribe: server-side transcription with `faster-whisper` (optional). Transcribed text is inserted into the main text area automatically.
- Edit transcript inline before analysis.
- NLP scoring (Sentiment, Confidence, Communication) and ML prediction (TF-IDF + Logistic Regression).
- Clean, mobile-friendly UI and exportable history (optional).

---

## Quick Start

### Prerequisites

- Python 3.9 or later
- `pip`

### Install & run

```bash
git clone https://github.com/yourusername/interview-analyzer.git
cd interview-analyzer
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

If you want offline transcription using Whisper (`faster-whisper`), install the optional packages:

```bash
pip install faster-whisper soundfile
```

Run the app:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (typically http://localhost:8501).

---

## Voice Input — How to use

1. Choose `Speak` mode in the UI and pick the language.
2. Click the recorder widget to start speaking (browser will ask microphone permission).
3. When done, click the pause/stop control on the recorder; the app saves the audio.
4. Click `Transcribe` to convert the saved audio to text (server-side). On success the text is automatically placed in the main text area for editing.
5. If you want to re-record from scratch, click `Record again` — the recorder resets and previous audio/transcript is cleared.

Notes:

- `st.audio_input` provides a simple browser-native recording experience — no server sound hardware or PortAudio is required.
- If `faster-whisper` is not installed, the Transcribe step will show an error explaining how to enable offline transcription.

---

## Analysis & Scoring (brief)

- NLP Score (weighted): Sentiment (30%), Confidence (40%), Communication (30%).
- ML Score: TF-IDF + Logistic Regression returns class probabilities and predicted label.
- Final score: weighted combination of NLP + ML scores, mapped to grade buckets (Excellent / Good / Fair / Needs Work).

---

## Model Training

To retrain the ML classifier:

```bash
python model/train_model.py
```

Training reads `data/training_data.csv` and writes model artifacts to `model/`.

---

## Troubleshooting

- If Streamlit isn't found: ensure your virtual environment is activated and `pip install -r requirements.txt` completed.
- If the browser recorder doesn't appear or microphone permission is denied: check browser permissions and try a different browser (Chrome/Edge/Firefox recommended).
- If transcription is noisy or slow: install `faster-whisper` and try a different Whisper model size (tiny → small → base) depending on CPU/RAM.

---

## Files of interest

- `app.py` — Streamlit UI and control flow (record → transcribe → edit → analyze)
- `utils/voice_input.py` — transcription helpers and model caching
- `model/train_model.py` — training pipeline and artifacts

---

## Requirements (core)

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
plotly>=5.20.0

Optional for offline transcription:
faster-whisper
soundfile
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit, push, and open a PR

Ideas: improve multi-language support, add CI checks, expand labelled dataset for the ML model.

---

## License

This project is licensed under MIT. See [LICENSE](LICENSE).

---

Built with Streamlit · scikit-learn · faster-whisper

⭐ Star the repo if it helped you
