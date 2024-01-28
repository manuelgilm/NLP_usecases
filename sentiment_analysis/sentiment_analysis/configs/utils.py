import pkgutil
from pathlib import Path

import yaml


def get_root_path() -> Path:
    """
    Get root path of the project
    """
    root_path = Path(__file__).parent.parent.parent
    return root_path


def get_config(path: str) -> dict:
    data = pkgutil.get_data(__name__, path)
    config = yaml.safe_load(data)
    return config


def read_txt(path: str) -> list:
    with open(path, "r") as f:
        data = f.readlines()

    return data
