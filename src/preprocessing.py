from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable

import pandas as pd


RANDOM_STATE = 42
TEXT_COLUMN = "text"
LABEL_COLUMN = "airline_sentiment"
AIRLINE_COLUMN = "airline"
NEGATIVE_REASON_COLUMN = "negativereason"
BINARY_LABELS = ["negative", "positive"]
KAGGLE_DATASET_HANDLE = "crowdflower/twitter-airline-sentiment"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def find_or_download_dataset(path: str | Path = "data/Tweets.csv") -> Path:
    """Use a local CSV when present, otherwise download the Kaggle dataset."""
    data_path = Path(path)
    if data_path.exists():
        return data_path

    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Add Tweets.csv there or install kagglehub."
        ) from exc

    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    matches = list(download_dir.rglob("Tweets.csv"))
    if not matches:
        raise FileNotFoundError(f"Tweets.csv was not found inside the Kaggle download: {download_dir}")
    return matches[0]


def load_airline_sentiment(path: str | Path = "data/Tweets.csv") -> pd.DataFrame:
    """Load and normalize the Twitter US Airline Sentiment dataset."""
    data_path = find_or_download_dataset(path)

    df = pd.read_csv(data_path)
    required = {TEXT_COLUMN, LABEL_COLUMN, AIRLINE_COLUMN}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    keep_columns = [
        column
        for column in [
            "tweet_id",
            AIRLINE_COLUMN,
            LABEL_COLUMN,
            "airline_sentiment_confidence",
            NEGATIVE_REASON_COLUMN,
            "negativereason_confidence",
            "retweet_count",
            TEXT_COLUMN,
            "tweet_created",
            "tweet_location",
        ]
        if column in df.columns
    ]
    return df[keep_columns].copy()


def clean_text(text: object, remove_stopwords: bool = True) -> str:
    """Basic tweet cleaning for transparent NLP preprocessing."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\brt\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text and simple length features used by EDA and models."""
    enriched = df.copy()
    enriched["clean_text"] = enriched[TEXT_COLUMN].apply(clean_text)
    enriched["tweet_length"] = enriched[TEXT_COLUMN].fillna("").str.len()
    enriched["word_count"] = enriched[TEXT_COLUMN].fillna("").str.split().str.len()
    return enriched


def binary_sentiment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only positive and negative examples for binary classification."""
    return df[df[LABEL_COLUMN].isin(BINARY_LABELS)].copy()


def balanced_binary_sentiment_rows(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return equal positive and negative samples using reproducible undersampling."""
    binary = binary_sentiment_rows(df)
    class_size = int(binary[LABEL_COLUMN].value_counts().min())
    balanced = (
        binary.groupby(LABEL_COLUMN, group_keys=False)
        .sample(n=class_size, random_state=random_state)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced


def tokenize_texts(texts: Iterable[str]) -> list[list[str]]:
    return [str(text).split() for text in texts]
