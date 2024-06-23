from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import mlflow
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from financial_news_sentiment.model.utils import get_or_create_experiment


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    prefix: Optional[str] = "",
    pos_label: Optional[str] = "positive",
    labels: Optional[List[str]] = ["positive", "negative"],
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :param prefix: Prefix for metric names.
    :return: Dictionary of classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, labels=labels, pos_label=pos_label
    )
    recall = recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def track_run(
    experiment_name: str,
    experiment_tags: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    run_tags: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    figures: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Track the run in MLflow.

    :param experiment_name: Name of the experiment.
    :param experiment_tags: Tags for the experiment.
    :param run_name: Name of the run.
    :param run_tags: Tags for the run.
    :param metrics: Dictionary of metrics.
    :param params: Dictionary of parameters.
    :param figures: Dictionary of figures
    :return: None
    """

    experiment_name = get_or_create_experiment(
        name=experiment_name, tags=experiment_tags
    )

    with mlflow.start_run(run_name=run_name, tags=run_tags):
        if metrics:
            mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)
        if figures:
            for name, figure in figures.items():
                mlflow.log_figure(figure, name)
