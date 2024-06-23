from typing import Any
from typing import Dict

import mlflow
from mlflow.entities.experiment import Experiment

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
