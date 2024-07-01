from pathlib import Path

from basic_chat.utils.utils import get_project_root


def test_get_project_root() -> Path:
    """
    Get the root directory of the project.
    """
    path = get_project_root()
    assert (
        path.as_posix() == "C:/Users/manue/Documents/NLP_usecases/basic_chat"
    )
