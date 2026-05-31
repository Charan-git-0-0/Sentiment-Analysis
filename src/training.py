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
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except Exception:
    tf = None

from src.evaluation import evaluate_predictions, save_json, save_model
from src.preprocessing import (
    BINARY_LABELS,
    LABEL_COLUMN,
    RANDOM_STATE,
    add_text_features,
    binary_sentiment_rows,
    load_airline_sentiment,
)


LABELS = BINARY_LABELS


def split_dataset(df: pd.DataFrame):
    enriched = add_text_features(binary_sentiment_rows(df))
    return train_test_split(
        enriched["clean_text"],
        enriched[LABEL_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=enriched[LABEL_COLUMN],
    )


def build_tfidf_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
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


def traditional_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": build_tfidf_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": build_tfidf_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = build_tfidf_pipeline(
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
        )
    return models


def train_traditional_models(dataset_path: str | Path = "data/Tweets.csv") -> pd.DataFrame:
    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)
    label_encoder = LabelEncoder().fit(LABELS)

    results = []
    for name, pipeline in traditional_models().items():
        results.append(_fit_traditional_model(name, pipeline, x_train, x_test, y_train, y_test, label_encoder))

    pd.DataFrame(results).to_csv("reports/model_results.csv", index=False)
    joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
    return pd.DataFrame(results)


def train_traditional_model(
    model_name: str,
    dataset_path: str | Path = "data/Tweets.csv",
) -> dict:
    """Train one TF-IDF model so each Colab notebook or script stays focused."""
    models = traditional_models()
    if model_name not in models:
        available = ", ".join(models)
        raise ValueError(f"Unknown or unavailable model '{model_name}'. Choose from: {available}")

    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)
    label_encoder = LabelEncoder().fit(LABELS)
    row = _fit_traditional_model(
        model_name,
        models[model_name],
        x_train,
        x_test,
        y_train,
        y_test,
        label_encoder,
    )
    joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
    _append_result(row)
    return row


def _fit_traditional_model(name, pipeline, x_train, x_test, y_train, y_test, label_encoder) -> dict:
    train_y = label_encoder.transform(y_train) if name == "XGBoost" else y_train
    start = time.perf_counter()
    pipeline.fit(x_train, train_y)
    elapsed = time.perf_counter() - start
    predictions = pipeline.predict(x_test)
    if name == "XGBoost":
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    metrics = evaluate_predictions(y_test, predictions, LABELS)
    save_model(pipeline, Path("saved_models") / f"{name.lower().replace(' ', '_')}.joblib")
    save_json(metrics, Path("reports") / f"{name.lower().replace(' ', '_')}_metrics.json")
    return _result_row(name, metrics, elapsed)


def _require_tensorflow():
    if tf is None:
        raise ImportError("TensorFlow is required for LSTM training. Install requirements.txt first.")


def train_lstm_model(
    dataset_path: str | Path = "data/Tweets.csv",
    max_words: int = 12000,
    max_len: int = 45,
    epochs: int = 8,
    batch_size: int = 64,
):
    _require_tensorflow()
    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)

    encoder = LabelEncoder().fit(LABELS)
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len, padding="post")
    test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_len, padding="post")

    model = Sequential(
        [
            Embedding(max_words, 96, input_length=max_len),
            LSTM(96, dropout=0.2, recurrent_dropout=0.1),
            Dense(64, activation="relu", name="feature_layer"),
            Dropout(0.25),
            Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.perf_counter()
    model.fit(
        train_seq,
        y_train_encoded,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    predictions = np.argmax(model.predict(test_seq, verbose=0), axis=1)
    pred_labels = encoder.inverse_transform(predictions)
    metrics = evaluate_predictions(y_test, pred_labels, LABELS)

    Path("saved_models").mkdir(exist_ok=True)
    model.save("saved_models/lstm_model.keras")
    joblib.dump(tokenizer, "saved_models/lstm_tokenizer.joblib")
    joblib.dump({"max_len": max_len, "labels": LABELS}, "saved_models/lstm_config.joblib")
    joblib.dump(encoder, "saved_models/label_encoder.joblib")
    save_json(metrics, "reports/lstm_metrics.json")
    _append_result(_result_row("LSTM", metrics, elapsed))
    return metrics


def train_lstm_svm(dataset_path: str | Path = "data/Tweets.csv"):
    _require_tensorflow()
    lstm_path = Path("saved_models/lstm_model.keras")
    if not lstm_path.exists():
        train_lstm_model(dataset_path)

    df = load_airline_sentiment(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(df)
    tokenizer = joblib.load("saved_models/lstm_tokenizer.joblib")
    config = joblib.load("saved_models/lstm_config.joblib")
    max_len = config["max_len"]
    encoder = LabelEncoder().fit(LABELS)

    train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len, padding="post")
    test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_len, padding="post")
    lstm = tf.keras.models.load_model(lstm_path)
    feature_model = Model(inputs=lstm.input, outputs=lstm.get_layer("feature_layer").output)

    train_features = feature_model.predict(train_seq, verbose=0)
    test_features = feature_model.predict(test_seq, verbose=0)

    svm = SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE)
    start = time.perf_counter()
    svm.fit(train_features, encoder.transform(y_train))
    elapsed = time.perf_counter() - start
    predictions = encoder.inverse_transform(svm.predict(test_features))
    metrics = evaluate_predictions(y_test, predictions, LABELS)

    save_model(svm, "saved_models/lstm_svm.joblib")
    save_json(metrics, "reports/lstm_svm_metrics.json")
    _append_result(_result_row("LSTM + SVM", metrics, elapsed))
    return metrics


def _result_row(name: str, metrics: dict, elapsed: float) -> dict:
    return {
        "Model": name,
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1": round(metrics["f1"], 4),
        "Training Time": round(elapsed, 2),
    }


def _append_result(row: dict) -> None:
    path = Path("reports/model_results.csv")
    if path.exists():
        results = pd.read_csv(path)
        results = results[results["Model"] != row["Model"]]
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    else:
        results = pd.DataFrame([row])
    results.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train airline sentiment models.")
    parser.add_argument("--data", default="data/Tweets.csv", help="Path to the airline sentiment CSV.")
    parser.add_argument(
        "--mode",
        choices=["traditional", "lstm", "hybrid", "all"],
        default="traditional",
        help="Training experiment to run.",
    )
    args = parser.parse_args()

    if args.mode in {"traditional", "all"}:
        train_traditional_models(args.data)
    if args.mode in {"lstm", "all"}:
        train_lstm_model(args.data)
    if args.mode in {"hybrid", "all"}:
        train_lstm_svm(args.data)


if __name__ == "__main__":
    main()
