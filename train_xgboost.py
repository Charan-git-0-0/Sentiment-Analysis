"""Train the TF-IDF + XGBoost comparison model."""

from src.training import train_traditional_model


if __name__ == "__main__":
    print(train_traditional_model("XGBoost"))
