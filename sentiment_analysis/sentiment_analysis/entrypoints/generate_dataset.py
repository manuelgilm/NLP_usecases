from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_analysis.configs.utils import get_config
from sentiment_analysis.configs.utils import get_root_path
from sentiment_analysis.data_preparation.retrieval import (
    generate_dataset,
)  # noqa
from sentiment_analysis.data_preparation.retrieval import (
    process_sentiment_file,
)  # noqa


def generate_dataset_from_raw_data():
    """
    Generate a dataset from raw data.
    """
    config = get_config("config.yaml")
    raw_data_path = config["DATA_FOLDER"]
    filenames = config["FILES"]
    resource_path = config["RESOURCES"]["RAW_DATASET"]
    root_path = get_root_path()

    data_dict_list = []
    for filename, filename_path in filenames.items():
        data_path = Path(raw_data_path) / filename_path
        print(f"Processing :{filename} ")
        sentiment_df = process_sentiment_file(data_path)
        data_dict_list.append(sentiment_df)

    df = generate_dataset(data_dict_list)

    df.to_csv(Path(root_path) / resource_path, index=False)


def generate_train_and_test_indexes():
    """
    Generate train and test data.
    """

    config = get_config("config.yaml")
    resource_path = config["RESOURCES"]["RAW_DATASET"]
    root_path = get_root_path()
    train_index_path = config["RESOURCES"]["TRAIN_INDEX"]
    test_index_path = config["RESOURCES"]["TEST_INDEX"]
    df = pd.read_csv(Path(root_path) / resource_path)
    train_df, test_df = train_test_split(
        df["id"], test_size=0.2, random_state=42
    )
    train_df.to_csv(Path(root_path) / train_index_path, index=False)
    test_df.to_csv(Path(root_path) / test_index_path, index=False)
