from pathlib import Path

import pandas as pd

from sentiment_analysis.configs.utils import get_config
from sentiment_analysis.configs.utils import get_root_path
from sentiment_analysis.configs.utils import read_txt
from sentiment_analysis.data_preparation.retrieval import (
    generate_dataset,
)  # noqa
from sentiment_analysis.data_preparation.retrieval import (
    process_sentiment_file,
)  # noqa
from sentiment_analysis.training.training_pipelines import (
    get_transformer_pipeline,
)  # noqa


def test_get_config():
    """
    Test get_config function.
    """
    config = get_config("config.yaml")
    path = config["DATA_FOLDER"]
    filenames = config["FILES"]
    data_path = Path(path) / filenames["IMDB"]
    data = read_txt(data_path)
    print(data)


def test_read_dataset() -> pd.DataFrame:
    """
    Test read_dataset function.
    """
    config = get_config("config.yaml")
    dataset_path = config["RESOURCES"]["RAW_DATASET"]
    df = pd.read_csv(Path(get_root_path()) / dataset_path)
    print(df.head())
    print(df.shape)


def process_raw_data():
    """
    Process raw data.
    """
    config = get_config("config.yaml")
    path = config["DATA_FOLDER"]
    filenames = config["FILES"]
    data_dict_list = []
    for filename, filename_path in filenames.items():
        data_path = Path(path) / filename_path
        print(f"Processing :{filename} ")
        sentiment_df = process_sentiment_file(data_path)
        data_dict_list.append(sentiment_df)
    df = generate_dataset(data_dict_list)
    print(df.head())
    print(df.shape)


def transform_data():
    """
    Transform data.
    """
    config = get_config("config.yaml")
    train_index_path = config["RESOURCES"]["TRAIN_INDEX"]
    test_index_path = config["RESOURCES"]["TEST_INDEX"]
    data_path = config["RESOURCES"]["RAW_DATASET"]
    df = pd.read_csv(Path(get_root_path()) / data_path)
    train_index = pd.read_csv(Path(get_root_path()) / train_index_path)
    test_index = pd.read_csv(Path(get_root_path()) / test_index_path)
    df_train = train_index.merge(df, on="id")
    df_test = test_index.merge(df, on="id")

    # transform labels to int
    df_train["label"] = df_train["label"].map(
        lambda x: 1 if x == "positive" else 0
    )  # noqa
    df_test["label"] = df_test["label"].map(
        lambda x: 1 if x == "positive" else 0
    )  # noqa

    text_columns = ["sentence"]

    pipeline = get_transformer_pipeline(text_columns=text_columns)
    pipeline.fit(df_train[text_columns], df_train["label"])

    y_pred = pipeline.predict(df_test[text_columns])
    predictions = pd.DataFrame(
        {
            "sentence": df_test["sentence"],
            "label": df_test["label"],
            "prediction": y_pred,
        }
    )
    print(predictions.head(10))
