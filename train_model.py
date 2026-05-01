"""
model/train_model.py
---------------------
MLA COMPONENT — Interview Answer Quality Classifier

Algorithm  : Logistic Regression (multi-class, one-vs-rest)
Features   : TF-IDF Vectorizer (unigrams + bigrams)
Classes    : Poor | Average | Good | Excellent
Dataset    : data/training_data.csv (120 labelled samples)

Run this ONCE before launching the app:
    python model/train_model.py

Outputs:
    model/tfidf_vectorizer.pkl
    model/logistic_regression_model.pkl
    model/label_encoder.pkl
    model/model_evaluation_report.txt
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model       import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics            import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline           import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data",  "training_data.csv")
MODEL_DIR  = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

VEC_PATH   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CLF_PATH   = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")
RPT_PATH   = os.path.join(MODEL_DIR, "model_evaluation_report.txt")

# Class order (worst → best)
CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["answer","label"])
    df["answer"] = df["answer"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    print(f"  Loaded {len(df)} samples")
    print(f"  Class distribution:")
    for cls in CLASS_ORDER:
        n = (df["label"] == cls).sum()
        print(f"    {cls:10s}: {n:3d} samples")
    return df


def build_pipeline():
    """
    TF-IDF + Logistic Regression pipeline.

    TF-IDF settings:
        ngram_range=(1,2)  — unigrams and bigrams capture phrases like 'I led'
        max_features=2000  — keeps top 2000 features by TF-IDF score
        sublinear_tf=True  — log normalization reduces impact of very frequent words
        min_df=2           — ignore terms appearing in fewer than 2 documents

    Logistic Regression:
        multi_class='ovr'  — one-vs-rest for 4-class problem
        C=1.0              — regularization strength
        max_iter=1000      — enough iterations to converge
        solver='lbfgs'     — efficient for small datasets
    """
    tfidf = TfidfVectorizer(
        ngram_range   = (1, 2),
        max_features  = 2000,
        sublinear_tf  = True,
        min_df        = 2,
        strip_accents = "unicode",
        lowercase     = True,
    )
    clf = LogisticRegression(
    C           = 1.0,
    max_iter    = 1000,
    solver      = "lbfgs",
    class_weight= "balanced",
    random_state= 42,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train_and_evaluate():
    print("\n" + "="*60)
    print("  MLA — Interview Quality Classifier Training")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    df = load_data()
    X  = df["answer"].tolist()
    y  = df["label"].tolist()

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(CLASS_ORDER)            # fit on fixed order so encoding is deterministic
    y_enc = le.transform(y)

    # ── Train/test split ──────────────────────────────────────────────────────
    print("\n[2] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test : {len(X_test)} samples")

    # ── Build + fit pipeline ──────────────────────────────────────────────────
    print("\n[3] Building TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Pipeline trained successfully")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n[4] Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"  Per-fold:    {[round(s,3) for s in cv_scores]}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[5] Evaluating on held-out test set...")
    y_pred     = pipeline.predict(X_test)
    test_acc   = accuracy_score(y_test, y_pred)
    class_names = le.inverse_transform(sorted(set(y_enc)))

    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_ORDER[i] for i in sorted(set(y_enc))],
        digits=3
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Test Accuracy: {test_acc:.3f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:")
    print(f"  (rows=actual, cols=predicted, order: {CLASS_ORDER})")
    print(f"  {cm}")

    # ── Top TF-IDF features per class ────────────────────────────────────────
    print("\n[6] Top 10 TF-IDF features per class:")
    tfidf_vec   = pipeline.named_steps["tfidf"]
    clf_model   = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf_vec.get_feature_names_out())

    for i, cls in enumerate(CLASS_ORDER):
        if i < len(clf_model.coef_):
            top_idx  = np.argsort(clf_model.coef_[i])[-10:][::-1]
            top_feat = feature_names[top_idx]
            print(f"  {cls:10s}: {list(top_feat)}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\n[7] Saving model artefacts...")

    # Save the full pipeline (vectorizer + model together)
    with open(CLF_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # Save vectorizer and label encoder separately for inspection
    with open(VEC_PATH, "wb") as f:
        pickle.dump(tfidf_vec, f)

    with open(ENC_PATH, "wb") as f:
        pickle.dump(le, f)

    # Save text evaluation report
    report_text = f"""
MLA INTERVIEW QUALITY CLASSIFIER — EVALUATION REPORT
======================================================
Algorithm  : Logistic Regression (One-vs-Rest)
Features   : TF-IDF Vectorizer (unigrams + bigrams, top 2000)
Classes    : {CLASS_ORDER}
Dataset    : {len(df)} labelled samples
Split      : 80% train / 20% test

CROSS-VALIDATION (5-fold stratified)
CV Accuracy : {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})
Per-fold    : {[round(s,3) for s in cv_scores]}

TEST SET RESULTS
Test Accuracy: {test_acc:.3f}

Classification Report:
{report}

Confusion Matrix (rows=actual, cols=predicted):
Classes: {CLASS_ORDER}
{cm}
"""
    with open(RPT_PATH, "w") as f:
        f.write(report_text)

    print(f"  Saved: {CLF_PATH}")
    print(f"  Saved: {VEC_PATH}")
    print(f"  Saved: {ENC_PATH}")
    print(f"  Saved: {RPT_PATH}")

    print("\n" + "="*60)
    print(f"  Training complete. Test accuracy: {test_acc:.1%}")
    print("  Run:  streamlit run app.py")
    print("="*60 + "\n")

    return pipeline, le, test_acc, cv_scores.mean()


if __name__ == "__main__":
    train_and_evaluate()
