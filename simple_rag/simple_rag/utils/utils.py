from pathlib import Path


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def get_test_data_dir():
    """
    Get the test data directory.
    """
    return get_project_root() / "test_data"
