import pkgutil
from pathlib import Path
from typing import List

import yaml


def get_root_path() -> Path:
    """
    Get root path of the project
    """
    root_path = Path(__file__).parent.parent.parent
    return root_path


def get_config(path: str) -> dict:
    """
    G3et config from yaml file.

    :param path: path to yaml file
    :return: config
    """
    data = pkgutil.get_data(__name__, path)
    config = yaml.safe_load(data)
    return config


def read_txt(path: str) -> List:
    """
    Read txt file.

    :param path: path to txt file
    :return: list of strings
    """
    with open(path, "r") as f:
        data = f.readlines()

    return data
