"""
model/predictor.py
-------------------
MLA COMPONENT — loads the trained LR + TF-IDF pipeline
and predicts answer quality class with confidence probabilities.

Usage:
    from model.predictor import predict_quality
    result = predict_quality("I led a team and delivered...")
"""

import os
import pickle

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CLF_PATH  = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
ENC_PATH  = os.path.join(MODEL_DIR, "label_encoder.pkl")

CLASS_ORDER  = ["Poor", "Average", "Good", "Excellent"]
CLASS_SCORES = {"Poor": 10, "Average": 35, "Good": 65, "Excellent": 90}

_pipeline     = None
_label_encoder = None


def _load_model():
    global _pipeline, _label_encoder

    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(
            f"Model not found at {CLF_PATH}.\n"
            "Run:  python model/train_model.py  first."
        )

    if _pipeline is None:
        with open(CLF_PATH, "rb") as f:
            _pipeline = pickle.load(f)

    if _label_encoder is None:
        with open(ENC_PATH, "rb") as f:
            _label_encoder = pickle.load(f)


def is_model_trained() -> bool:
    """Check whether the saved model files exist."""
    return os.path.exists(CLF_PATH) and os.path.exists(ENC_PATH)


def predict_quality(text: str) -> dict:
    """
    Predict the quality class of an interview answer.

    Returns:
        {
            "predicted_class" : str    e.g. "Good"
            "ml_score"        : int    mapped score 0-100
            "probabilities"   : dict   {class: probability}
            "confidence"      : float  probability of predicted class
            "reasoning"       : str    human-readable explanation
        }
    """
    _load_model()

    # Predict class and probability distribution
    proba_array = _pipeline.predict_proba([text])[0]

    # Map probabilities to class names via label encoder
    classes_encoded = _pipeline.classes_
    classes_named   = _label_encoder.inverse_transform(classes_encoded)

    proba_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes_named, proba_array)
    }

    # Predicted class = highest probability
    predicted_class = max(proba_dict, key=proba_dict.get)
    confidence      = proba_dict[predicted_class]
    ml_score        = CLASS_SCORES.get(predicted_class, 50)

    # Weighted score across all probabilities
    weighted_score = sum(
        CLASS_SCORES.get(cls, 0) * prob
        for cls, prob in proba_dict.items()
    )
    ml_score = int(round(weighted_score))

    # Human-readable reasoning
    reasoning = _build_reasoning(predicted_class, confidence, proba_dict)

    return {
        "predicted_class" : predicted_class,
        "ml_score"        : ml_score,
        "probabilities"   : {c: proba_dict.get(c, 0.0) for c in CLASS_ORDER},
        "confidence"      : round(confidence, 3),
        "reasoning"       : reasoning,
    }


def _build_reasoning(predicted: str, confidence: float, proba: dict) -> str:
    conf_pct = int(confidence * 100)
    second_class = sorted(proba, key=proba.get, reverse=True)[1]
    second_pct   = int(proba[second_class] * 100)

    lines = {
        "Excellent": (
            f"The model classifies this as an Excellent answer ({conf_pct}% confidence). "
            f"The TF-IDF features strongly match patterns from high-scoring training samples — "
            f"specific achievements, assertive language, and measurable outcomes."
        ),
        "Good": (
            f"The model classifies this as a Good answer ({conf_pct}% confidence). "
            f"The answer contains relevant keywords and reasonable structure, "
            f"though it lacks the specific quantified achievements seen in Excellent answers."
        ),
        "Average": (
            f"The model classifies this as an Average answer ({conf_pct}% confidence). "
            f"The TF-IDF features are present but weak — the answer lacks concrete evidence "
            f"and the language is generic. The second most likely class is {second_class} ({second_pct}%)."
        ),
        "Poor": (
            f"The model classifies this as a Poor answer ({conf_pct}% confidence). "
            f"The TF-IDF features match patterns associated with filler words, "
            f"negative framing, and lack of specific content."
        ),
    }
    return lines.get(predicted, f"Predicted: {predicted} with {conf_pct}% confidence.")
