"""
utils/voice_input.py
---------------------
Speech-to-text using faster-whisper (OpenAI Whisper, fully offline).
Records audio via sounddevice, transcribes with Whisper tiny model.

Install:
    pip install faster-whisper sounddevice scipy numpy

No internet needed. No API key. Works on Mac, Windows, Linux.
"""

import io
import threading
import numpy as np

# ── Lazy imports so app doesn't crash if libs missing ─────────────────────────
def _check_imports():
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    return missing


def is_microphone_available() -> bool:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return any(d["max_input_channels"] > 0 for d in devices)
    except Exception:
        return False


def list_microphones() -> list:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [d["name"] for d in devices if d["max_input_channels"] > 0]
    except Exception:
        return []


# ── Whisper model — loaded once and cached ─────────────────────────────────────
_whisper_model = None

def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # "tiny" model: ~75MB download, fast, good accuracy for interviews
        # Change to "base" for slightly better accuracy (larger download)
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def _record_audio(duration: int, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone into a numpy array."""
    import sounddevice as sd
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()   # blocks until recording is done
    return audio.flatten()


def _record_audio_with_stop(
    duration: int,
    sample_rate: int,
    stop_flag: threading.Event,
    chunk_seconds: int = 2,
) -> np.ndarray:
    """
    Record in small chunks so stop_flag is checked frequently.
    Returns all recorded audio concatenated.
    """
    import sounddevice as sd
    chunks = []
    elapsed = 0

    while elapsed < duration:
        if stop_flag and stop_flag.is_set():
            break
        remaining = min(chunk_seconds, duration - elapsed)
        chunk = sd.rec(
            int(remaining * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        chunks.append(chunk.flatten())
        elapsed += remaining

    if not chunks:
        return np.array([], dtype="float32")
    return np.concatenate(chunks)


def _transcribe_audio(audio: np.ndarray, language: str = None) -> str:
    """Run Whisper transcription on a numpy float32 array."""
    if audio is None or len(audio) == 0:
        return ""

    model = _get_model()

    # Whisper expects float32 numpy array at 16kHz
    # language=None → auto-detect (works for Hindi, English, etc.)
    # language="en" → force English (faster)
    lang = None
    if language:
        # Map BCP-47 to Whisper language codes
        mapping = {
            "en-IN": "en",
            "en-US": "en",
            "en-GB": "en",
            "hi-IN": "hi",
        }
        lang = mapping.get(language, None)

    segments, _ = model.transcribe(
        audio,
        language=lang,
        beam_size=3,
        vad_filter=True,           # skip silent parts automatically
        vad_parameters=dict(
            min_silence_duration_ms=500
        ),
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION — live streaming transcription
# ═══════════════════════════════════════════════════════════════════════════════

def record_and_transcribe_live(
    duration: int = 30,
    language: str = "en-IN",
    text_placeholder=None,
    stop_flag: threading.Event = None,
) -> dict:
    """
    Record audio in chunks, transcribe each chunk with Whisper,
    and stream the growing transcript to a Streamlit placeholder live.

    Args:
        duration         : max total seconds to listen
        language         : BCP-47 code e.g. "en-IN", "hi-IN", "en-US"
        text_placeholder : st.empty() — updated live after each chunk
        stop_flag        : threading.Event — set to stop early

    Returns:
        {"success": bool, "text": str, "error": str}
    """
    # Check dependencies
    missing = _check_imports()
    if missing:
        return {
            "success": False,
            "text": "",
            "error": f"Missing libraries: {', '.join(missing)}. Run: pip install {' '.join(missing)}"
        }

    def _show(text: str, recording: bool = True):
        if text_placeholder is None:
            return
        cursor = (
            '<span style="display:inline-block;width:8px;height:14px;'
            'background:#4A6FA5;margin-left:3px;vertical-align:middle;'
            'animation:blink 1s infinite;"></span>'
            '<style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}</style>'
        ) if recording else ""
        display = text if text.strip() else ("Listening..." if recording else "")
        text_placeholder.markdown(
            f"""
            <div style="
                background:#fff; border:1px solid #D8D4CE; border-radius:6px;
                padding:1rem 1.1rem; font-family:'DM Sans',sans-serif;
                font-size:0.95rem; color:#1a1a1a; line-height:1.65; min-height:90px;
            ">{display}{cursor}</div>
            """,
            unsafe_allow_html=True,
        )

    import sounddevice as sd

    SAMPLE_RATE  = 16000
    CHUNK_SECS   = 5      # transcribe every 5 seconds — good live feel
    full_parts   = []

    try:
        _show("", recording=True)
        elapsed = 0

        while elapsed < duration:
            if stop_flag and stop_flag.is_set():
                break

            chunk_len = min(CHUNK_SECS, duration - elapsed)
            if chunk_len <= 0:
                break

            # Record one chunk
            audio_chunk = sd.rec(
                int(chunk_len * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            elapsed += chunk_len

            if stop_flag and stop_flag.is_set():
                # Still transcribe what we captured before stop
                chunk_text = _transcribe_audio(audio_chunk.flatten(), language)
                if chunk_text:
                    full_parts.append(chunk_text)
                break

            # Transcribe this chunk
            chunk_text = _transcribe_audio(audio_chunk.flatten(), language)
            if chunk_text:
                full_parts.append(chunk_text)

            # Update live display
            current = " ".join(full_parts)
            _show(current, recording=(elapsed < duration))

    except Exception as e:
        return {"success": False, "text": "", "error": f"Recording error: {str(e)}"}

    final_text = " ".join(full_parts).strip()
    _show(final_text, recording=False)

    if not final_text:
        return {
            "success": False,
            "text": "",
            "error": (
                "No speech detected. Make sure your microphone is not muted "
                "and speak clearly within 30cm of it."
            ),
        }

    return {"success": True, "text": final_text, "error": ""}


# ── Simple one-shot version (kept for backward compatibility) ──────────────────
def record_and_transcribe(duration: int = 20, language: str = "en-IN") -> dict:
    return record_and_transcribe_live(duration=duration, language=language)