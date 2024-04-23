import pkgutil
from pathlib import Path
from typing import Any
from typing import Dict

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
