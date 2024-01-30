from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd

from sentiment_analysis.sentiment_analysis.utils.utils import get_config
from sentiment_analysis.sentiment_analysis.utils.utils import get_root_path


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
    data_df = data_df.reset_index()
    data_df = data_df.rename(columns={"index": "id"})
    data_df["id"] = data_df.index
    return data_df


def get_train_and_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate train and test data.

    :return: train and test data.
    """
    config = get_config("config.yaml")
    train_index_path = config["RESOURCES"]["TRAIN_INDEX"]
    test_index_path = config["RESOURCES"]["TEST_INDEX"]
    data_path = config["RESOURCES"]["RAW_DATASET"]
    # Reading data
    df = pd.read_csv(Path(get_root_path()) / data_path)
    train_index = pd.read_csv(Path(get_root_path()) / train_index_path)
    test_index = pd.read_csv(Path(get_root_path()) / test_index_path)
    # Merging data
    df_train = train_index.merge(df, on="id")
    df_test = test_index.merge(df, on="id")

    return df_train, df_test
