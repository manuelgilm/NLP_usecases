from typing import Dict
from typing import Optional

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# fmt: off
from sentiment_analysis.training.training_pipelines import get_transformer_pipeline  # noqa

# fmt: on


def objective_function(
    params: Dict[str, float],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    experiment_id: str,
) -> float:
    """
    Objective function to optimize.

    :param params: hyperparameters to evaluate.
    :param x_train: train features.
    :param y_train: train labels.
    :param x_val: validation features.
    :param y_val: validation labels.
    :param experiment_id: mlflow experiment id.
    :return: score to minimize.
    """

    # get model:
    model = get_transformer_pipeline(text_columns=["sentence"])
    # cast params to int:
    params["classifier__max_depth"] = int(params["classifier__max_depth"])
    params["classifier__n_estimators"] = int(
        params["classifier__n_estimators"]
    )
    # set model params:
    model.set_params(**params)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        metrics = log_classification_metrics(y_pred, y_val, run.info.run_id)

    return -metrics["f1"]


def log_classification_metrics(
    y_pred: pd.DataFrame, y_true: pd.DataFrame, run_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Get classification metrics and log them to mlflow.

    :param y_pred: predicted labels.
    :param y_true: true labels.
    :param run_id: mlflow run id.
    :return: classification metrics.
    """
    if not run_id:
        run_id = mlflow.active_run().info.run_id
        if not run_id:
            raise Exception("No mlflow Run was found!")
    with mlflow.start_run(run_id=run_id):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }
        mlflow.log_metrics(metrics)
        return metrics
