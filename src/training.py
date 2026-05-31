from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError:
    tf = None

from src.evaluation import evaluate_predictions, save_json, save_model
from src.preprocessing import (
    BINARY_LABELS,
    LABEL_COLUMN,
    RANDOM_STATE,
    add_text_features,
    balanced_binary_sentiment_rows,
    load_airline_sentiment,
)


LABELS = BINARY_LABELS
DEFAULT_DATASET = "data/Tweets.csv"
TRADITIONAL_MODEL_NAMES = ["logistic", "random-forest", "xgboost"]
SUPPORTED_RESULTS = {"Logistic Regression", "Random Forest", "XGBoost", "LSTM"}


def prepare_folders() -> None:
    Path("reports").mkdir(exist_ok=True)
    Path("saved_models").mkdir(exist_ok=True)


def split_dataset(df: pd.DataFrame):
    """Create the same balanced, reproducible 80/20 split for every model."""
    balanced = balanced_binary_sentiment_rows(df)
    enriched = add_text_features(balanced)
    return train_test_split(
        enriched["clean_text"],
        enriched[LABEL_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=enriched[LABEL_COLUMN],
    )


def build_tfidf_pipeline(classifier) -> Pipeline:
    """Convert cleaned tweets into unigram and bigram TF-IDF features."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=12000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("model", classifier),
        ]
    )






"""Logistic REgression, Random Forest, and XGBoost all use the same TF-IDF features, so we can avoid duplicating that setup code."""




def build_traditional_model(model_name: str) -> tuple[str, Pipeline]:
    """Build one traditional classifier without duplicating TF-IDF setup."""
    if model_name == "logistic":
        return (
            "Logistic Regression",
            build_tfidf_pipeline(
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            ),
        )

    if model_name == "random-forest":
        return (
            "Random Forest",
            build_tfidf_pipeline(
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            ),
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Run `pip install xgboost`.")
        return (
            "XGBoost",
            build_tfidf_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=250,
                    learning_rate=0.08,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=RANDOM_STATE,
                )
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def train_traditional_model(model_name: str, dataset_path: str = DEFAULT_DATASET) -> dict:
    """Train, evaluate, and save one TF-IDF model."""
    prepare_folders()
    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)
    display_name, pipeline = build_traditional_model(model_name)

    encoder = LabelEncoder().fit(LABELS)
    training_labels = encoder.transform(y_train) if model_name == "xgboost" else y_train

    start = time.perf_counter()
    pipeline.fit(x_train, training_labels)
    elapsed = time.perf_counter() - start

    predictions = pipeline.predict(x_test)
    if model_name == "xgboost":
        predictions = encoder.inverse_transform(predictions.astype(int))

    metrics = evaluate_predictions(y_test, predictions, LABELS)
    file_stem = display_name.lower().replace(" ", "_")
    save_model(pipeline, f"saved_models/{file_stem}.joblib")
    save_json(metrics, f"reports/{file_stem}_metrics.json")
    joblib.dump(encoder, "saved_models/label_encoder.joblib")

    result = result_row(display_name, metrics, elapsed)
    update_comparison_table(result)
    return result




'''LSTM models require a different setup, so they get their own training function. The traditional models are all trained on TF-IDF features, but the LSTM learns an embedding directly from the text.'''



def train_lstm_model(
    dataset_path: str = DEFAULT_DATASET,
    max_words: int = 12000,
    max_len: int = 45,
    epochs: int = 8,
    batch_size: int = 64,
) -> dict:
    """Train a sequence model that learns an embedding directly from the tweets."""
    if tf is None:
        raise ImportError("TensorFlow is not installed. Run `pip install tensorflow`.")

    prepare_folders()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)
    encoder = LabelEncoder().fit(LABELS)
    y_train_encoded = encoder.transform(y_train)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    train_sequences = pad_sequences(
        tokenizer.texts_to_sequences(x_train),
        maxlen=max_len,
        padding="post",
    )
    test_sequences = pad_sequences(
        tokenizer.texts_to_sequences(x_test),
        maxlen=max_len,
        padding="post",
    )

    model = Sequential(
        [
            tf.keras.Input(shape=(max_len,), dtype="int32"),
            Embedding(max_words, 96, mask_zero=True),
            LSTM(96, dropout=0.2),
            Dense(64, activation="relu"),
            Dropout(0.25),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    start = time.perf_counter()
    model.fit(
        train_sequences,
        y_train_encoded,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    probabilities = model.predict(test_sequences, verbose=0).ravel()
    predictions = encoder.inverse_transform((probabilities >= 0.5).astype(int))
    metrics = evaluate_predictions(y_test, predictions, LABELS)

    model.save("saved_models/lstm_model.keras")
    joblib.dump(tokenizer, "saved_models/lstm_tokenizer.joblib")
    joblib.dump({"max_len": max_len, "labels": LABELS}, "saved_models/lstm_config.joblib")
    joblib.dump(encoder, "saved_models/label_encoder.joblib")
    save_json(metrics, "reports/lstm_metrics.json")

    result = result_row("LSTM", metrics, elapsed)
    update_comparison_table(result)
    return result


def result_row(model_name: str, metrics: dict, elapsed: float) -> dict:
    return {
        "Model": model_name,
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1": round(metrics["f1"], 4),
        "Training Time": round(elapsed, 2),
    }


def update_comparison_table(result: dict) -> None:
    """Insert or replace one model row without erasing other experiment results."""
    path = Path("reports/model_results.csv")
    if path.exists():
        results = pd.read_csv(path)
        results = results[results["Model"].isin(SUPPORTED_RESULTS)]
        results = results[results["Model"] != result["Model"]]
        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
    else:
        results = pd.DataFrame([result])
    results.to_csv(path, index=False)


def train_all(dataset_path: str) -> None:
    for model_name in TRADITIONAL_MODEL_NAMES:
        print(train_traditional_model(model_name, dataset_path))
    print(train_lstm_model(dataset_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train airline sentiment models.")
    parser.add_argument(
        "--model",
        choices=[*TRADITIONAL_MODEL_NAMES, "lstm", "traditional", "all"],
        default="traditional",
        help="Model or group of models to train.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATASET,
        help="Optional CSV path. KaggleHub downloads the dataset when this file is absent.",
    )
    args = parser.parse_args()

    if args.model == "traditional":
        for model_name in TRADITIONAL_MODEL_NAMES:
            print(train_traditional_model(model_name, args.data))
    elif args.model == "all":
        train_all(args.data)
    elif args.model == "lstm":
        print(train_lstm_model(args.data))
    else:
        print(train_traditional_model(args.model, args.data))


if __name__ == "__main__":
    main()
