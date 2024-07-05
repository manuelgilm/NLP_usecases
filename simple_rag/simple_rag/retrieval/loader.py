from langchain_community.document_loaders import PyPDFLoader

from simple_rag.utils.utils import get_test_data_dir

test_data = (
    get_test_data_dir() / "Building Python Microservices with FastAPI.pdf"
)


def get_pdf_data():
    """
    Get the pdf data.
    """
    loader = PyPDFLoader(test_data)
    pages = loader.load_and_split()
    print(pages[0])
