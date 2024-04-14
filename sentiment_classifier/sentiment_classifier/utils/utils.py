from pathlib import Path

import spacy


def get_root_dir() -> Path:
    """
    Get root directory of the project.

    :return: Root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def load_spacy_model(model_path: str):
    """
    Load spacy model from path.

    :param model_path: Path to spacy model.
    """

    root_dir = get_root_dir()
    path = root_dir / model_path
    nlp = spacy.load(path)
    return nlp
