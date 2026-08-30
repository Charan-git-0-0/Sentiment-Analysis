# TweetLens — Airline Twitter Sentiment Analytics Dashboard ✈️📊

An end-to-end Natural Language Processing (NLP) pipeline and interactive Streamlit analytics platform that classifies airline customer feedback, benchmarks traditional Machine Learning baselines against Deep Learning (LSTM), and visualizes real-time sentiment distributions and KPIs.

---

## 🚀 Key Highlights

- **NLP Preprocessing Pipeline**: Tokenization, lemmatization, stop-word filtering, and balanced sampling on 30.1K+ airline tweet records.
- **Model Benchmarking**:
  - **Logistic Regression (TF-IDF)**: Ultra-fast 0.14s CPU inference with strong linear baseline performance.
  - **Tree Ensembles**: Random Forest & XGBoost evaluating non-linear feature interactions.
  - **Deep Learning**: Recurrent Neural Network (LSTM) with learned word embeddings and dynamic padding masking (mask_zero=True).
- **Interactive Streamlit Dashboard**: Live inference evaluator, confusion matrix visualizer, negative reason breakdown, and sentiment trends.

---

## 🛠️ Model Comparison Matrix

| Model | Feature Representation | Key Advantage | Target Metric |
|---|---|---|---|
| **Logistic Regression** | TF-IDF (Unigrams + Bigrams) | Ultra-fast inference (<0.15s), highly interpretable | Baseline Accuracy |
| **Random Forest** | TF-IDF (Unigrams + Bigrams) | Robust bagging ensemble, resilient to overfitting | Ensemble Comparison |
| **XGBoost** | TF-IDF (Unigrams + Bigrams) | Gradient-boosted decision trees | Non-linear Split Optimization |
| **LSTM (RNN)** | Learned Dense Word Embeddings | Sequential word context & long-range dependencies | High-Recall Sentiment Detection |

---

## 📂 Project Structure

`	ext
Sentiment-Analysis/
├── app.py                  # Streamlit analytics dashboard & live predictor
├── src/
│   ├── preprocessing.py    # Text cleaning, regex normalization, TF-IDF vectorizer
│   ├── training.py         # Model training & checkpointing routines
│   ├── evaluation.py       # Classification report, ROC-AUC & confusion matrix
│   ├── visualization.py    # Matplotlib / Seaborn chart generation
│   └── prediction.py       # Inference pipeline for single & batch inputs
├── notebooks/              # Exploratory Data Analysis & experiments
├── saved_models/           # Serialized model weights & vectorizer artifacts
├── reports/                # Evaluation metrics JSON and summary CSVs
├── requirements.txt        # Python package dependencies
├── .gitignore
└── README.md
`

---

## ⚡ Quick Start Guide

### 1. Clone & Install Dependencies
`ash
git clone https://github.com/Charan-git-0-0/Sentiment-Analysis.git
cd Sentiment-Analysis

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
`

### 2. Train Models
`ash
# Train all models (Logistic Regression, Random Forest, XGBoost, LSTM)
python -m src.training --model all
`

### 3. Launch Streamlit Analytics Dashboard
`ash
streamlit run app.py
`
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📄 License
MIT © Charan-git-0-0