"""Extract LSTM features and train the optional SVM classifier."""

from src.training import train_lstm_svm


if __name__ == "__main__":
    print(train_lstm_svm())
