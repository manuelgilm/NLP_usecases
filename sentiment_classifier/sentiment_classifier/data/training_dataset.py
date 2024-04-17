from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
import spacy
from spacy.tokens import DocBin

from sentiment_classifier.data.retrieval import get_train_test_data
from sentiment_classifier.data.retrieval import read_csv
from sentiment_classifier.utils.utils import get_root_dir
from sentiment_classifier.utils.utils import read_config


def get_spacy_pipeline(model: str):  # review the return type
    """
    This function takes in a model name and returns a spacy pipeline.

    :param model: Name of the spacy model.
    :return: Spacy pipeline.
    """
    nlp = spacy.load(model)
    return nlp


def get_list_of_tuples(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Get list of tuples from a dataframe.

    :param df: dataframe
    :return: list of tuples
    """
    df["tuples"] = df.apply(
        lambda row: (row["text"], row["sentiment"]), axis=1
    )
    return df["tuples"].tolist()


def create_spacy_documents(
    nlp: spacy.language, data: List[Tuple[str, str]]
) -> List[spacy.tokens.Doc]:
    """
    This function takes in a list of tuples and returns a list of
    spacy documents.
    """
    text = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if label == "positive":
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 0
        elif label == "negative":
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
            doc.cats["neutral"] = 0
        else:
            doc.cats["positive"] = 0
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 1

        text.append(doc)
    return text


def create_spacy_dataset(
    nlp, data: List[Tuple[str, str]], path: Union[Path, str]
) -> None:
    """
    Create a spacy dataset from a given data.

    :param data: List of tuples.
    :param path: Path to save the dataset.
    :return: None
    """

    if isinstance(path, str):
        path = Path(path)
    print("Creating spacy documents...")
    docs = create_spacy_documents(nlp, data)
    print("Spacy documents created.")

    print("Creating spacy binary dataset...")
    doc_bin = DocBin(docs=docs)
    print("Spacy binary dataset created.")

    folder = path.parent
    if not folder.exists():
        folder.mkdir(parents=True)

    doc_bin.to_disk(path)


def create_training_and_testing_dataset():
    """
    Create training and testing dataset for the sentiment analysis model.
    """
    configs = read_config("data_config")
    root = get_root_dir()
    data_path = root / configs["source_data"]
    df = read_csv(
        data_path, encoding="latin-1", header=None, names=["sentiment", "text"]
    )

    train, test = get_train_test_data(df)
    train_data = get_list_of_tuples(train)
    test_data = get_list_of_tuples(test)

    # transform the data
    model_name = "en_core_web_trf"
    nlp = get_spacy_pipeline(model=model_name)
    # save the data
    train_path = root / configs["binary_train_data"]
    test_path = root / configs["binary_test_data"]

    create_spacy_dataset(nlp, train_data, train_path)
    create_spacy_dataset(nlp, test_data, test_path)
