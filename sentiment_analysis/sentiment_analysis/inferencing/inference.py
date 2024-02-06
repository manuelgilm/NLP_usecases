from typing import Any
from typing import Dict

import mlflow
import pandas as pd


def get_prediction(config: Dict[str, Any], data: pd.DataFrame):
    """
    Get prediction from pipeline.

    :param pipeline: sklearn pipeline
    :param data: data to predict
    :return: prediction
    """

    run_id = config["RUN_ID"]
    pipeline = load_model(run_id)
    prediction = pipeline.predict(data)

    return prediction


def load_model(run_id: str):
    """
    Load model from mlflow.

    :param run_id: mlflow run id
    :return: sklearn pipeline
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model
