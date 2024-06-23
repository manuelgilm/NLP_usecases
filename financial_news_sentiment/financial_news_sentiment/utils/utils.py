import pkgutil
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
import yaml


def get_root_path():
    """
    Get the root path of the project.
    """
    return Path(__file__).parent.parent.parent


def read_config(name: str) -> Dict[str, Any]:
    """
    Read a yaml configuration file.

    :param name: Name of the configuration file.
    :return: Configuration dictionary.
    """
    try:
        data = pkgutil.get_data(
            "financial_news_sentiment", f"configs/{name}.yaml"
        )
    except Exception as e:
        raise FileNotFoundError(f"Config file not found: {name}.yaml") from e

    config = yaml.safe_load(data)
    return config


def add_encoding(path: Union[str, Path]) -> str:
    """
    Add encoding to path.

    :param path: Path.
    :return: Path with encoding.
    """
    if isinstance(path, str):
        path = Path(path)

    name_without_suffix = "".join(path.name.split(".")[:-1])
    new_suffix = str(time.time()).split(".")[0]
    new_name = f"{name_without_suffix}_{new_suffix}{path.suffix}"
    return path.parent / new_name


def decode_path(path: Union[str, Path]) -> str:
    """
    Decode path.

    :param path: Path.
    :return: Decoded path.
    """

    if isinstance(path, str):
        path = Path(path)

    name = path.name.split(".")[0]
    timestamp = name.split("_")[-1]
    date_ = datetime.fromtimestamp(float(timestamp), timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return date_


def as_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.

    :param data: List of dictionaries.
    :return: DataFrame.
    """
    keys = data[0].keys()
    data_ = {key: [] for key in keys}
    for d in data:
        for key in keys:
            data_[key].append(d[key])

    return pd.DataFrame(data_)
