from pathlib import Path

from sentiment_analysis.configs.utils import get_config
from sentiment_analysis.configs.utils import read_txt
from sentiment_analysis.data_preparation.retrieval import generate_dataset
from sentiment_analysis.data_preparation.retrieval import process_sentiment_file  # noqa


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
