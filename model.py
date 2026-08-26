import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def build_model():
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
        activation='relu',                  # ReLU activation
        solver='adam',                      # Adam optimizer
        alpha=0.001,                        # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )

    print("✅ Model built: MLP Deep Neural Network")
    print("   Architecture : Input → 256 → 128 → 64 → Output")
    print("   Activation   : ReLU (hidden) + Logistic (output)")
    print("   Optimizer    : Adam")
    print("   Regularizer  : L2 (alpha=0.001)")

    return model


def train_model(model, X_train, y_train):
    print("\n⏳ Training the neural network...")
    model.fit(X_train, y_train)
    print(f"✅ Training complete!")
    print(f"   Epochs run  : {model.n_iter_}")
    print(f"   Final loss  : {model.loss_:.4f}")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print("  MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print("="*50)
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (Ham→Ham)   : {cm[0][0]}")
    print(f"  False Positives (Ham→Spam)  : {cm[0][1]}")
    print(f"  False Negatives (Spam→Ham)  : {cm[1][0]}")
    print(f"  True Positives  (Spam→Spam) : {cm[1][1]}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    return {
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1,
        'confusion_matrix': cm, 'y_pred': y_pred
    }


def predict_email(model, vectorizer, email_text):
    import re, string

    text = email_text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    features = vectorizer.transform([text]).toarray()
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = proba[pred]

    label = "🚨 SPAM" if pred == 1 else "✅ HAM"
    return label, confidence


if __name__ == "__main__":
    from dataset import create_dataset
    from preprocessing import preprocess_dataframe, extract_features, split_data

    df = create_dataset()
    df = preprocess_dataframe(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)