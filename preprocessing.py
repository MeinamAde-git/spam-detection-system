import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataframe(df):
    df = df.copy()
    df['cleaned_email'] = df['email'].apply(clean_text)
    print("✅ Text preprocessing complete.")
    print(f"   Sample raw    : {df['email'].iloc[0][:60]}...")
    print(f"   Sample cleaned: {df['cleaned_email'].iloc[0][:60]}...")
    return df


def extract_features(df, max_features=3000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df['cleaned_email']).toarray()
    y = df['label'].values

    print(f"\n✅ Feature extraction complete.")
    print(f"   Feature matrix shape : {X.shape}")
    print(f"   Vocabulary size      : {len(vectorizer.vocabulary_)}")

    return X, y, vectorizer


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\n✅ Data split complete.")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Test samples     : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from dataset import create_dataset

    df = create_dataset()
    df = preprocess_dataframe(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)