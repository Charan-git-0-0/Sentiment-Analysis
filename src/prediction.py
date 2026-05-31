from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import clean_text


DEFAULT_MODEL_PATH = Path("saved_models/logistic_regression.joblib")
LABEL_ENCODER_PATH = Path("saved_models/label_encoder.joblib")
LIVE_MODEL_FILES = {
    "logistic_regression.joblib",
    "random_forest.joblib",
    "xgboost.joblib",
    "lstm_model.keras",
}
MODEL_DISPLAY_NAMES = {
    "logistic_regression.joblib": "Logistic Regression",
    "random_forest.joblib": "Random Forest",
    "xgboost.joblib": "XGBoost",
    "lstm_model.keras": "LSTM",
}



def available_model_paths() -> list[Path]:
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return []
    return sorted(path for path in model_dir.iterdir() if path.name in LIVE_MODEL_FILES)


def available_models() -> list[str]:
    return [MODEL_DISPLAY_NAMES[path.name] for path in available_model_paths()]


def predict_sentiment(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.training` after adding data/Tweets.csv."
        )

    cleaned = clean_text(text)
    if model_path.suffix == ".keras":
        prediction, confidence = _predict_with_lstm(cleaned, model_path)
    else:
        prediction, confidence = _predict_with_traditional_model(cleaned, model_path)

    return {
        "text": text,
        "clean_text": cleaned,
        "sentiment": str(prediction),
        "confidence": confidence,
        "model": MODEL_DISPLAY_NAMES.get(model_path.name, model_path.stem.replace("_", " ").title()),
    }


def _predict_with_traditional_model(cleaned: str, model_path: Path):

    model = joblib.load(model_path)
    prediction = model.predict([cleaned])[0]
    if isinstance(prediction, (int, np.integer)) and LABEL_ENCODER_PATH.exists():
        encoder = joblib.load(LABEL_ENCODER_PATH)
        prediction = encoder.inverse_transform([int(prediction)])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(np.max(probabilities))
    return prediction, confidence


def _predict_with_lstm(cleaned: str, model_path: Path):
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError as exc:
        raise ImportError("TensorFlow is required to use the saved LSTM model.") from exc

    tokenizer = joblib.load("saved_models/lstm_tokenizer.joblib")
    config = joblib.load("saved_models/lstm_config.joblib")
    encoder = joblib.load(LABEL_ENCODER_PATH)
    sequence = pad_sequences(
        tokenizer.texts_to_sequences([cleaned]),
        maxlen=config["max_len"],
        padding="post",
    )
    model = tf.keras.models.load_model(model_path)
    positive_probability = float(model.predict(sequence, verbose=0).ravel()[0])
    encoded_prediction = int(positive_probability >= 0.5)
    prediction = encoder.inverse_transform([encoded_prediction])[0]
    confidence = max(positive_probability, 1 - positive_probability)
    return prediction, confidence
