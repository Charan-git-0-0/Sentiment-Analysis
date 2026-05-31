from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation import load_json, load_results
from src.prediction import available_model_paths, predict_sentiment
from src.preprocessing import add_text_features, load_airline_sentiment
from src.visualization import (
    airline_sentiment_distribution,
    airline_sentiment_heatmap,
    dataset_overview,
    missing_values,
    negative_reason_distribution,
    sentiment_distribution,
    tweet_length_histogram,
    word_frequencies,
    wordcloud_image,
)


st.set_page_config(
    page_title="Airline Twitter Sentiment Analytics",
    page_icon="A",
    layout="wide",
)

DATA_PATH = Path("data/Tweets.csv")
RESULTS_PATH = Path("reports/model_results.csv")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return add_text_features(load_airline_sentiment(DATA_PATH))


def render_missing_data_message() -> None:
    st.warning("The airline sentiment dataset could not be loaded.")
    st.markdown(
        "Add `data/Tweets.csv` or install the compatible KaggleHub dependency from `requirements.txt`. "
        "Expected columns include `text`, `airline_sentiment`, `airline`, and `negativereason`."
    )


def analytics_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    metric_cols = st.columns(5)
    overview = dataset_overview(df)
    for index, row in overview.iterrows():
        metric_cols[index % 5].metric(row["Metric"], row["Value"])

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(sentiment_distribution(df), use_container_width=True)
    with right:
        st.plotly_chart(airline_sentiment_distribution(df), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        reason_fig = negative_reason_distribution(df)
        if reason_fig:
            st.plotly_chart(reason_fig, use_container_width=True)
    with right:
        st.plotly_chart(tweet_length_histogram(df), use_container_width=True)

    st.plotly_chart(airline_sentiment_heatmap(df), use_container_width=True)

    st.subheader("Frequent Words")
    tabs = st.tabs(["All Tweets", "Positive", "Negative", "Missing Values"])
    for tab, sentiment in zip(tabs[:3], [None, "positive", "negative"]):
        with tab:
            words = word_frequencies(df, sentiment=sentiment)
            left, right = st.columns([1, 1.2])
            with left:
                st.dataframe(words, use_container_width=True, hide_index=True)
            with right:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_image(df, sentiment=sentiment), interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig, clear_figure=True)
    with tabs[3]:
        missing = missing_values(df)
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)


def model_performance_dashboard() -> None:
    results = load_results(RESULTS_PATH)
    if results.empty:
        st.info("No model results yet. Run `python -m src.training` after adding `data/Tweets.csv`.")
        return

    st.subheader("Model Comparison")
    st.dataframe(results, use_container_width=True, hide_index=True)

    metric = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1"])
    fig = px.bar(results, x="Model", y=metric, text=metric, title=f"{metric} Comparison")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrices")
    metric_files = [
        Path("reports") / name
        for name in [
            "logistic_regression_metrics.json",
            "random_forest_metrics.json",
            "xgboost_metrics.json",
            "lstm_metrics.json",
        ]
        if (Path("reports") / name).exists()
    ]
    if not metric_files:
        st.caption("Confusion matrices will appear after model training.")
        return

    labels = ["negative", "positive"]
    selected = st.selectbox("Model metrics file", metric_files, format_func=lambda p: p.stem.replace("_metrics", "").replace("_", " ").title())
    metrics = load_json(selected)
    if "confusion_matrix" in metrics:
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        heatmap = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        st.plotly_chart(heatmap, use_container_width=True)


def live_prediction_dashboard() -> None:
    st.subheader("Live Sentiment Prediction")
    model_files = available_model_paths()
    if not model_files:
        st.info("Train a traditional ML model first with `python -m src.training`.")
        return

    selected_model = st.selectbox(
        "Model",
        model_files,
        format_func=lambda p: p.stem.replace("_model", "").replace("_", " ").title(),
    )
    tweet = st.text_area(
        "Tweet",
        value="This airline lost my luggage and customer service was terrible.",
        height=120,
    )
    if st.button("Predict", type="primary") and tweet.strip():
        result = predict_sentiment(tweet, selected_model)
        st.metric("Predicted Sentiment", result["sentiment"].title())
        if result["confidence"] is not None:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        st.caption(f"Model used: {result['model']}")
        st.text_input("Cleaned tweet", value=result["clean_text"], disabled=True)


st.title("Airline Twitter Sentiment Analytics Dashboard")
st.caption("Business analytics, traditional ML, deep learning, and transparent NLP on airline customer feedback.")

page = st.sidebar.radio(
    "Dashboard",
    ["Analytics Dashboard", "Model Performance Dashboard", "Live Prediction Dashboard"],
)

if page == "Analytics Dashboard":
    try:
        analytics_dashboard(load_data())
    except (FileNotFoundError, ImportError, ValueError):
        render_missing_data_message()
elif page == "Model Performance Dashboard":
    model_performance_dashboard()
else:
    live_prediction_dashboard()
