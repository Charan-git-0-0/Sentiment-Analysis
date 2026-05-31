from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

from src.preprocessing import AIRLINE_COLUMN, LABEL_COLUMN, NEGATIVE_REASON_COLUMN


SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c2410c",
    "neutral": "#64748b",
    "positive": "#15803d",
}


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Airlines": df[AIRLINE_COLUMN].nunique() if AIRLINE_COLUMN in df else 0,
        "Sentiment classes": df[LABEL_COLUMN].nunique() if LABEL_COLUMN in df else 0,
        "Missing values": int(df.isna().sum().sum()),
    }
    return pd.DataFrame(rows.items(), columns=["Metric", "Value"])


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    return missing.reset_index().rename(columns={"index": "column", 0: "missing_count"})


def sentiment_distribution(df: pd.DataFrame):
    counts = df[LABEL_COLUMN].value_counts().reindex(SENTIMENT_ORDER).dropna().reset_index()
    counts.columns = ["sentiment", "count"]
    return px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        text="count",
        title="Sentiment Distribution",
    )


def airline_sentiment_distribution(df: pd.DataFrame):
    grouped = (
        df.groupby([AIRLINE_COLUMN, LABEL_COLUMN])
        .size()
        .reset_index(name="count")
        .sort_values([AIRLINE_COLUMN, LABEL_COLUMN])
    )
    return px.bar(
        grouped,
        x=AIRLINE_COLUMN,
        y="count",
        color=LABEL_COLUMN,
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        title="Airline-wise Sentiment Distribution",
    )


def negative_reason_distribution(df: pd.DataFrame):
    if NEGATIVE_REASON_COLUMN not in df.columns:
        return None
    reasons = (
        df[df[LABEL_COLUMN] == "negative"][NEGATIVE_REASON_COLUMN]
        .dropna()
        .value_counts()
        .reset_index()
    )
    reasons.columns = ["negative_reason", "count"]
    return px.bar(
        reasons,
        x="count",
        y="negative_reason",
        orientation="h",
        text="count",
        title="Negative Reason Analysis",
    )


def tweet_length_histogram(df: pd.DataFrame):
    return px.histogram(
        df,
        x="tweet_length",
        color=LABEL_COLUMN,
        nbins=40,
        marginal="box",
        color_discrete_map=SENTIMENT_COLORS,
        title="Tweet Length Distribution",
    )


def airline_sentiment_heatmap(df: pd.DataFrame):
    table = pd.crosstab(df[AIRLINE_COLUMN], df[LABEL_COLUMN], normalize="index")
    table = table.reindex(columns=[c for c in SENTIMENT_ORDER if c in table.columns])
    return px.imshow(
        table,
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Sentiment Share by Airline",
    )


def word_frequencies(df: pd.DataFrame, sentiment: str | None = None, top_n: int = 25) -> pd.DataFrame:
    source = df
    if sentiment:
        source = df[df[LABEL_COLUMN] == sentiment]
    counter: Counter[str] = Counter()
    for text in source["clean_text"].dropna():
        counter.update(str(text).split())
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def wordcloud_image(df: pd.DataFrame, sentiment: str | None = None):
    source = df
    if sentiment:
        source = df[df[LABEL_COLUMN] == sentiment]
    corpus = " ".join(source["clean_text"].dropna().astype(str))
    if not corpus.strip():
        corpus = "no words available"
    return WordCloud(width=1000, height=500, background_color="white", colormap="viridis").generate(corpus)
