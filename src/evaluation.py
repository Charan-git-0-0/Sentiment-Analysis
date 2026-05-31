from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


SUPPORTED_MODELS = {"Logistic Regression", "Random Forest", "XGBoost", "LSTM"}


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_json(payload: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_model(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_results(path: str | Path = "reports/model_results.csv") -> pd.DataFrame:
    result_path = Path(path)
    if result_path.exists():
        results = pd.read_csv(result_path)
        return results[results["Model"].isin(SUPPORTED_MODELS)]
    return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Training Time"])
