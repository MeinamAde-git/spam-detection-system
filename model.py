import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from dataset import create_multiclass_dataset


def train_and_save_multiclass_model():
    os.makedirs("outputs", exist_ok=True)

    print("1. Generating multi-class dataset...")
    df = create_multiclass_dataset(n_samples=3000)

    X = df["text"]
    y_raw = df["category"]

    # Encode string labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("3. Training Multi-Layer Perceptron (MLP)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42
    )
    mlp.fit(X_train_vec, y_train)

    print("4. Evaluating Model...")
    y_pred = mlp.predict(X_test_vec)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

    model_payload = {
        "model": mlp,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "classes": encoder.classes_.tolist()
    }

    output_path = os.path.join("outputs", "multiclass_spam_model.pkl")
    joblib.dump(model_payload, output_path)
    print(f"\nModel pipeline saved successfully to '{output_path}'!")


if __name__ == "__main__":
    train_and_save_multiclass_model()