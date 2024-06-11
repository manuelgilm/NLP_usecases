from typing import List

import numpy as np
import pandas as pd
from transformers import pipeline

from financial_news_sentiment.data.retrieval import get_train_test_data
from financial_news_sentiment.data.retrieval import read_data
from financial_news_sentiment.utils.utils import add_encoding
from financial_news_sentiment.utils.utils import get_root_path


def get_prediction(texts: List[str], labels: List[str] = None):
    """
    Get sentiment analysis predictions for a list of texts.

    :param texts: List of texts.
    :return: List of sentiment analysis predictions.
    """

    sentiment_analysis = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )
    return sentiment_analysis(texts, candidate_labels=labels)


def predict():
    """
    Get sentiment analysis prediction for a single text.

    :param text: Text.
    :return: Sentiment analysis prediction.
    """
    root = get_root_path()

    df = read_data()
    labels = df["sentiment"].unique().tolist()
    train_df, _ = get_train_test_data(df)
    texts = train_df["text"].iloc[0:10].tolist()
    print(train_df)
    predictions = get_prediction(texts, labels)
    result = {
        "text": texts,
        "sentiment": train_df["sentiment"].iloc[0:10].tolist(),
        "prediction": [
            p["labels"][np.argmax(p["scores"])] for p in predictions
        ],
        "score": [p["scores"][np.argmax(p["scores"])] for p in predictions],
    }
    result_df = pd.DataFrame(result)
    output_path = (
        root
        / "financial_news_sentiment"
        / "output"
        / "predictions_all_data.csv"
    )
    output_path = add_encoding(output_path)
    result_df.to_csv(output_path, index=False)
    print(result_df)
