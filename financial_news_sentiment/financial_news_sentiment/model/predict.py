from typing import List

import pandas as pd
from transformers import pipeline

from financial_news_sentiment.data.retrieval import get_train_test_data
from financial_news_sentiment.data.retrieval import read_data


def get_prediction(texts: List[str]):
    """
    Get sentiment analysis predictions for a list of texts.

    :param texts: List of texts.
    :return: List of sentiment analysis predictions.
    """

    sentiment_analysis = pipeline("sentiment-analysis")
    return sentiment_analysis(texts)


def predict():
    """
    Get sentiment analysis prediction for a single text.

    :param text: Text.
    :return: Sentiment analysis prediction.
    """

    df = read_data()
    train_df, _ = get_train_test_data(df)

    texts = train_df["text"].iloc[0:4].tolist()
    predictions = get_prediction(texts)
    result = {
        "text": texts,
        "sentiment": train_df["sentiment"].iloc[0:4].tolist(),
        "prediction": [p["label"] for p in predictions],
        "score": [p["score"] for p in predictions],
    }
    result_df = pd.DataFrame(result)
    print(result_df)
