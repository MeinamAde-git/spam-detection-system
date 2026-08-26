"""
================================================
SPAM EMAIL DETECTION USING DEEP LEARNING
Step 5: Main Runner — Complete Pipeline
================================================
Run this file to execute everything at once!
"""

import os
import pickle

from dataset import create_dataset
from preprocessing import preprocess_dataframe, extract_features, split_data
from model import build_model, train_model, evaluate_model, predict_email
from visualizations import generate_all_plots

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        SPAM EMAIL DETECTION USING DEEP LEARNING             ║
║        Acad$emic Project  |  Python + MLP Neural Network     ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Test Emails for Live Demo ─────────────────────────────────────────────────
TEST_EMAILS = [
    "Congratulations! You've won a $1,000,000 lottery! Click here to claim now!",
    "Hi, can we reschedule our meeting to 3 PM on Friday? Let me know.",
    "FREE money!! Limited time offer. Send your details to receive cash reward.",
    "Please find the invoice attached. Payment is due by end of this month.",
    "URGENT: Your bank account has been compromised. Click here immediately!",
]


def main():
    print(BANNER)

    # ── STEP 1: Load Dataset ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    df = create_dataset()

    # ── STEP 2: Preprocess Text ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Text Preprocessing")
    print("=" * 60)
    df = preprocess_dataframe(df)

    # ── STEP 3: Feature Extraction ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Feature Extraction (TF-IDF)")
    print("=" * 60)
    X, y, vectorizer = extract_features(df)

    # ── STEP 4: Train / Test Split ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Train / Test Split (80% / 20%)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── STEP 5: Build Model ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Building Deep Neural Network")
    print("=" * 60)
    model = build_model()

    # ── STEP 6: Train Model ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Training the Model")
    print("=" * 60)
    model = train_model(model, X_train, y_train)

    # ── STEP 7: Evaluate Model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Evaluating the Model")
    print("=" * 60)
    metrics = evaluate_model(model, X_test, y_test)

    # ── STEP 8: Visualizations ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Generating Visualizations")
    print("=" * 60)
    generate_all_plots(df, model, vectorizer, metrics)

    # ── STEP 9: Live Predictions ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Sample Predictions on New Emails")
    print("=" * 60)
    for i, email in enumerate(TEST_EMAILS, 1):
        label, confidence = predict_email(model, vectorizer, email)
        preview = email[:60] + "..." if len(email) > 60 else email
        print(f"\nEmail {i}: \"{preview}\"")
        print(f"  → Prediction : {label}")
        print(f"  → Confidence : {confidence*100:.1f}%")

    # ── Save Model ─────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/spam_model.pkl", "wb") as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
    print("\n✅ Model saved to outputs/spam_model.pkl")

    print("\n" + "=" * 60)
    print("  🎉 PROJECT COMPLETE!")
    print("  All outputs saved in the 'outputs/' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()