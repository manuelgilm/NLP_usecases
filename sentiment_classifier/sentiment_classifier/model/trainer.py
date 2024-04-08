import spacy 
from spacy.tokens import DocBin
from datetime import datetime
from pathlib import Path
from typing import List
from typing import Tuple 
from typing import Union 

def get_spacy_pipeline(model:str)->spacy.lang.en.English: # review the return type
    """
    This function takes in a model name and returns a spacy pipeline.

    :param model: Name of the spacy model.
    :return: Spacy pipeline.
    """
    nlp = spacy.load(model)
    return nlp


def create_spacy_documents(nlp:spacy.language, data:List[Tuple[str, str]])->List[spacy.Doc]:
    """
    This function takes in a list of tuples and returns a list of spacy documents.
    """
    text = []
    for doc, label in nlp.pipe(data, as_tuples = True):
        if (label=='positive'):
            doc.cats['positive'] = 1
            doc.cats['negative'] = 0
            doc.cats['neutral']  = 0
        elif (label=='negative'):
            doc.cats['positive'] = 0
            doc.cats['negative'] = 1
            doc.cats['neutral']  = 0
        else:
            doc.cats['positive'] = 0
            doc.cats['negative'] = 0
            doc.cats['neutral']  = 1

        text.append(doc)
    return(text)

def create_spacy_dataset(data:List[Tuple[str, str]], path:Union[Path, str])->None:
    """
    Create a spacy dataset from a given data.

    :param data: List of tuples.
    :param path: Path to save the dataset.
    :return: None
    """

    if isinstance(path, str):
        path = Path(path)
    print("Creating spacy documents...")
    docs = create_spacy_documents(data)
    print("Spacy documents created.")

    print("Creating spacy binary dataset...")
    doc_bin = DocBin(docs = docs)
    print("Spacy binary dataset created.")

    folder = path.parent
    if not folder.exists():
        folder.mkdir(parents=True)
        
    doc_bin.to_disk(path)


