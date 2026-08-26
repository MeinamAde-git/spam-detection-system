import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset import create_multiclass_dataset


def train_and_benchmark_models():
    os.makedirs("outputs", exist_ok=True)

    print("1. Generating dataset...")
    df = create_multiclass_dataset(n_samples=3000)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])
    X = df["text"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Vectorizing text...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=50,
                                              random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
    }

    benchmark_metrics = []

    print("\n3. Training & Benchmarking Architectures...")
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = (time.time() - t0) * 1000

        t0_inf = time.time()
        y_pred = clf.predict(X_test_vec)
        inf_time = (time.time() - t0_inf) * 1000 / len(y_test)

        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average="weighted") * 100

        benchmark_metrics.append({
            "Model": name,
            "Accuracy (%)": round(acc, 2),
            "F1-Score (%)": round(f1, 2),
            "Train Time (ms)": round(train_time, 2),
            "Inference Latency (ms/sample)": round(inf_time, 4)
        })
        print(f" -> {name}: Accuracy={acc:.2f}%, Latency={inf_time:.4f}ms/sample")

    benchmark_df = pd.DataFrame(benchmark_metrics)

    payload = {
        "models": models,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "classes": encoder.classes_.tolist(),
        "benchmarks": benchmark_df
    }

    output_path = os.path.join("outputs", "multiclass_spam_model.pkl")
    joblib.dump(payload, output_path)
    print(f"\nSaved all 3 models to '{output_path}'!")
    print("\nBenchmark Summary:\n", benchmark_df.to_string(index=False))


if __name__ == "__main__":
    train_and_benchmark_models()