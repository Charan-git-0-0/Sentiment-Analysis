from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import clean_text


DEFAULT_MODEL_PATH = Path("saved_models/logistic_regression.joblib")
LABEL_ENCODER_PATH = Path("saved_models/label_encoder.joblib")


def available_models() -> list[str]:
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return []
    return sorted(path.stem.replace("_", " ").title() for path in model_dir.glob("*.joblib") if "encoder" not in path.stem)


def predict_sentiment(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.training` after adding data/Tweets.csv."
        )

    model = joblib.load(model_path)
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    if isinstance(prediction, (int, np.integer)) and LABEL_ENCODER_PATH.exists():
        encoder = joblib.load(LABEL_ENCODER_PATH)
        prediction = encoder.inverse_transform([int(prediction)])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(np.max(probabilities))
    return {
        "text": text,
        "clean_text": cleaned,
        "sentiment": str(prediction),
        "confidence": confidence,
        "model": model_path.stem.replace("_", " ").title(),
    }
