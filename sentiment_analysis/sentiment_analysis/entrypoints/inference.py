import argparse

import pandas as pd

# fmt: off
from sentiment_analysis.data_preparation.retrieval import get_train_and_test_data  # noqa
from sentiment_analysis.inferencing.inference import get_prediction
from sentiment_analysis.utils.utils import get_config

# fmt: on


def inference():
    """
    Inference pipeline.
    """
    config = get_config("config.yaml")
    _, test_df = get_train_and_test_data()
    prediction = get_prediction(config=config, data=test_df)
    return prediction


def classify_text():
    """
    Classify text.
    """

    parser = argparse.ArgumentParser(description="Classify text")
    parser.add_argument("--text", type=str, help="Text to classify")
    args = parser.parse_args()
    text = args.text

    config = get_config("config.yaml")
    df = pd.DataFrame([text], columns=["sentence"])
    prediction = get_prediction(config=config, data=df)
    prediction = "positive" if prediction[0] == 1 else "negative"
    return prediction
