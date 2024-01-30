import mlflow

from sentiment_analysis.utils.utils import get_root_path


def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Get or create mlflow experiment.

    :param experiment_name: name of the experiment.
    :return: experiment_id
    """

    artifact_path = get_root_path() / "mlruns"
    mlflow.set_tracking_uri(artifact_path.as_uri())
    client = mlflow.tracking.MlflowClient()
    try:
        experiment_id = client.create_experiment(experiment_name)
    except BaseException:
        experiment_id = client.get_experiment_by_name(
            experiment_name
        ).experiment_id
    finally:
        mlflow.set_experiment(experiment_name)

    return experiment_id
