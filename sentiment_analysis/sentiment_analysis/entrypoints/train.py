from functools import partial

import mlflow
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import tpe
from sklearn.model_selection import train_test_split

# fmt: off
from sentiment_analysis.data_preparation.retrieval import get_train_and_test_data  # noqa
from sentiment_analysis.training.mlflow_utils import get_or_create_mlflow_experiment  # noqa
from sentiment_analysis.training.optimize import objective_function
from sentiment_analysis.utils.utils import get_config

# fmt: on


def train():
    """
    Train a machine learning model.
    """

    config = get_config("config.yaml")
    experiment_name = config["MLFLOW"]["EXPERIMENT_NAME"]
    # get train and test data:
    train_df, test_df = get_train_and_test_data()

    # get or create mlflow experiment:
    experiment_id = get_or_create_mlflow_experiment(experiment_name)

    search_space = config["HYPERPARAMETERS"]["RANDOMFOREST"]

    # split data into train and validation:
    x_train, x_val, y_train, y_val = train_test_split(
        train_df.drop(["label"], axis=1),
        train_df["label"],
        test_size=0.2,
        random_state=42,
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID: {run.info.run_id}")
        best_params = fmin(
            fn=partial(
                objective_function,
                x_train,
                y_train,
                x_val,
                y_val,
                experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
            show_progressbar=True,
        )

    return best_params
