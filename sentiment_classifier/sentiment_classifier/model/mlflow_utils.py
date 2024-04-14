from typing import Any
from typing import Dict

import mlflow
from mlflow.entities import Experiment

from sentiment_classifier.utils.utils import get_root_dir


def get_or_create_experiment(
    experiment_name: str, tags: Dict[str, Any]
) -> Experiment:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment.
    :return: Experiment
    """

    root_dir = get_root_dir()

    mlflow.set_tracking_uri(root_dir / "mlruns")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, tags=tags
        )
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment
