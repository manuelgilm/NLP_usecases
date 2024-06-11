from typing import Any
from typing import Dict

import mlflow
from mlflow.entities import Experiment

from financial_news_sentiment.utils.utils import get_root_path


def get_or_create_experiment(name: str, tags: Dict[str, Any]) -> Experiment:
    """
    Get or create an experiment in MLflow.

    :param name: Name of the experiment.
    :param tags: Tags for the experiment.
    :return: Experiment.
    """

    root_dir = get_root_path()
    tracking_uri = (root_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(name, tags=tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(name)

    return experiment


def track_experiment(
    model,
    experiment_name: str,
    tags: Dict[str, Any],
    params: Dict[str, Any],
    metrics: Dict[str, Any],
):
    """
    Track an experiment in MLflow.

    :param experiment_name: Name of the experiment.
    :param tags: Tags for the experiment.
    :param params: Parameters for the experiment.
    :param metrics: Metrics for the experiment.
    """
    experiment = get_or_create_experiment(experiment_name, tags)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


# def mlflow_experiment(experiment_name: str, tags: Dict[str, Any]):
#     """
#     Decorator for MLflow experiment.

#     :param experiment_name: Name of the experiment.
#     :param tags: Tags for the experiment.
#     :return: Decorator.
#     """

#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             experiment = get_or_create_experiment(experiment_name, tags)
#             with mlflow.start_run(experiment_id=experiment.experiment_id):
#                 return func(*args, **kwargs)

#         return wrapper

#     return decorator
