from simple_rag.utils.utils import get_project_root


def test_get_project_root():
    """
    Test the get_project_root function.
    """
    assert get_project_root().name == "simple_rag"
    assert (
        get_project_root().as_posix()
        == "C:/Users/manue/Documents/NLP_usecases/simple_rag"
    )
