from pathlib import Path
from typing import Tuple
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_classifier.utils.utils import get_root_dir


def read_csv(path: Union[Path, str], **kwargs) -> pd.DataFrame:
    """
    Read a csv file from a given path and return a pandas dataframe.

    :param path: Path to the csv file.
    :return: Pandas dataframe.
    """
    if path is None:
        raise ValueError("Path cannot be None.")

    if not path.exists():
        raise FileNotFoundError(f"File not found at path: {path}")

    if path.suffix != ".csv":
        raise ValueError("File must be a csv file.")

    if isinstance(path, str):
        path = Path(path)

    return pd.read_csv(path, **kwargs)


def get_train_test_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get train and test data from a given dataframe.

    :param df: Pandas dataframe.
    :param test_size: Size of the test data.
    :param random_state: Random state for train_test_split.
    :return: Tuple of train and test dataframes.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=test_size,
        random_state=random_state,
    )

    train_df = pd.DataFrame({"text": X_train, "sentiment": y_train})
    test_df = pd.DataFrame({"text": X_test, "sentiment": y_test})

    return train_df, test_df
