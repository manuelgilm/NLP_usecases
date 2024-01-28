from typing import Dict
from typing import List

import pandas as pd


def process_sentiment_file(file_path: str) -> Dict[str, List[str]]:
    """
    Process sentiment file and return a Dictionary with two keys:
    sentence and label

    :param file_path: path to the sentiment file
    :return: Dictionary with two keys: sentence and label
    """
    sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                sentence, score = parts
                label = "positive" if float(score) > 0 else "negative"
                sentences.append(sentence)
                labels.append(label)

    data = {"sentence": sentences, "label": labels}
    return data


def generate_dataset(data: List[Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Generate a dataframe from a list of dictionaries with two keys:
    sentence and label

    :param data: List of Dictionaries with two keys: sentence and label
    :return: Pandas dataframe
    """
    # concatenate dictionaries:
    data_dict = {
        "sentence": [v for d in data for v in d["sentence"]],
        "label": [v for d in data for v in d["label"]],
    }
    # convert to dataframe:
    data_df = pd.DataFrame(data_dict)
    return data_df
