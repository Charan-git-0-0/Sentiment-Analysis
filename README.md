# Airline Twitter Sentiment Analytics Dashboard

An end-to-end NLP project that classifies airline tweets as positive or negative, compares traditional machine learning with an LSTM, and presents customer-feedback insights in a Streamlit dashboard.

The project uses the Twitter US Airline Sentiment dataset. It is designed to run on a laptop or in Google Colab and reports only metrics produced by your own training runs.

## What the Project Does

```text
Kaggle dataset
-> Load and clean tweets
-> Remove neutral rows for model training
-> Balance positive and negative examples
-> Create an 80/20 stratified train-test split
-> Train and evaluate models
-> Save metrics and model files
-> Display results in Streamlit
```

The dashboard can still analyze all tweets, including neutral rows.

## Dataset

The project first looks for:

```text
data/Tweets.csv
```

If the file is missing, the loader uses `kagglehub` to download:

```text
crowdflower/twitter-airline-sentiment
```

Training keeps all positive tweets and reproducibly samples the same number of negative tweets. This gives both classes equal weight without changing the original analytics dataset.

## Models

| Model | Features | Purpose |
| --- | --- | --- |
| Logistic Regression | TF-IDF unigrams and bigrams | Fast, strong baseline |
| Random Forest | TF-IDF unigrams and bigrams | Bagging-based tree comparison |
| XGBoost | TF-IDF unigrams and bigrams | Boosting comparison |
| LSTM | Learned embedding and word sequence | Deep-learning comparison |

## Project Structure

```text
project/
|-- app.py
|-- data/
|-- notebooks/
|-- models/
|-- src/
|   |-- preprocessing.py
|   |-- training.py
|   |-- evaluation.py
|   |-- visualization.py
|   `-- prediction.py
|-- saved_models/
|-- reports/
|-- screenshots/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Local Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train Models

Train the three traditional ML models:

```powershell
python -m src.training --model traditional
```

Train one model at a time:

```powershell
python -m src.training --model logistic
python -m src.training --model random-forest
python -m src.training --model xgboost
python -m src.training --model lstm
```

Train every model:

```powershell
python -m src.training --model all
```

## Google Colab

Upload the project folder or clone its Git repository, then run:

```python
!pip install -r requirements.txt
!python -m src.training --model all
```

KaggleHub downloads the dataset automatically if `data/Tweets.csv` is absent.

## Dashboard

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard provides:

- Dataset analytics
- Sentiment and airline comparisons
- Negative-reason analysis
- Word clouds and frequent words
- Model metric comparisons
- Confusion matrices
- Live predictions using saved traditional ML models or the trained LSTM

## Saved Results

Training updates:

```text
reports/model_results.csv
```

Each model also writes its own detailed JSON metrics:

```text
reports/logistic_regression_metrics.json
reports/random_forest_metrics.json
reports/xgboost_metrics.json
reports/lstm_metrics.json
```

Saved model files are written to:

```text
saved_models/
```

These files are binary artifacts loaded by Python and are not intended to be opened in a text editor.

## Notes

- TF-IDF often performs very well for short tweets because keywords and short phrases carry strong sentiment signals.
- The LSTM uses `mask_zero=True`, so padding added to short tweets is ignored.
- A more complex model is not assumed to be better. Compare the actual metrics and training times.
