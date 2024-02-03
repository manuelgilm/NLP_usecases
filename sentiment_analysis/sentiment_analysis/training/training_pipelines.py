from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_transformer_pipeline(model_prefix: str, text_columns: List[str]):
    """
    This function returns a sklearn pipeline with a ColumnTransformer.

    :param text_columns: List of text columns.
    :return: sklearn pipeline
    """
    text_transformer = ColumnTransformer(
        transformers=[
            (f"tfidf_{text_column}", TfidfVectorizer(), text_column)
            for text_column in text_columns
        ],
    )

    pipeline = Pipeline(
        steps=[
            ("text_processor", text_transformer),
            (model_prefix, RandomForestClassifier()),
        ]
    )
    return pipeline
