from pathlib import Path
import pkgutil
import spacy
import yaml

from typing import Any
from typing import Dict

def get_root_dir() -> Path:
    """
    Get root directory of the project.

    :return: Root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def read_config(name: str)-> Dict[str, Any]:
    """
    Read the configuration file.

    :param name: The name of the configuration file.
    """
    data = pkgutil.get_data("sentiment_classifier", f"configs/{name}.yaml")
    config = yaml.safe_load(data)
    return config

def load_spacy_model(model_path: str):
    """
    Load spacy model from path.

    :param model_path: Path to spacy model.
    """

    root_dir = get_root_dir()
    path = root_dir / model_path
    nlp = spacy.load(path)
    return nlp
