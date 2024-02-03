from typing import Any
from typing import Dict
from typing import Optional

import mlflow
import pandas as pd

# fmt: on
from hyperopt import hp
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# fmt: off
from sentiment_analysis.training.training_pipelines import get_transformer_pipeline  # noqa


def cast_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cast hyperparameters to the right type.

    :param params: hyperparameters to cast.
    :return: casted hyperparameters.
    """
    for param, value in params.items():
        if isinstance(value, float):
            params[param] = int(value)
        elif isinstance(value, int):
            params[param] = int(value)
        elif isinstance(value, str):
            params[param] = str(value)

    return params


def objective_function(
    params: Dict[str, float],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    experiment_id: str,
    model_prefix: str,
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
    model = get_transformer_pipeline(model_prefix=model_prefix,
                                     text_columns=["sentence"])
    params = cast_params(params)
    # set model params:
    model.set_params(**params)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        metrics = log_classification_metrics(y_pred, y_val, run.info.run_id)

    return -metrics["f1"]


def get_search_space(prefix: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get search space for hyperopt.

    :param prefix: prefix for the model.
    :param params: hyperparameters to optimize.
    :return: search space.
    """
    search_space = {}
    for param, options in params.items():
        param_label = prefix + "__" + param

        if options["type"] == "int":
            search_space[param_label] = hp.uniformint(
                param_label,
                int(options["min"]),
                int(options["max"])
            )
        elif options["type"] == "float":
            search_space[param_label] = hp.uniform(
                param_label,
                float(options["min"]),
                float(options["max"])
            )
        elif options["type"] == "str":
            search_space[param_label] = hp.choice(
                param_label,
                options["options"]
            )

    return search_space


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
            raise Exception("No Active MLflow Run was found!")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    mlflow.log_metrics(metrics)
    return metrics
