from sentiment_classifier.utils.utils import get_root_dir
from sentiment_classifier.model.mlflow_utils import get_or_create_experiment
from sentiment_classifier.utils.utils import load_spacy_model
from spacy.cli.train import train
from typing import Union 
from typing import Optional
import mlflow 
from mlflow.models.signature import infer_signature
from pathlib import Path


def train_spacy_model(config_path:Optional[str]=None, training_data_path:Optional[str]=None, testing_data_path:Optional[str]=None, output_path:Optional[str]=None)->None:
    """
    Train a spacy model.

    :param model_path: Path to the spacy model.
    :param training_data_path: Path to the training data.
    :param output_path: Path to save the model.
    :param n_iter: Number of iterations.
    :return: None
    """
    root_dir = get_root_dir()
    if config_path is None:
        config_path = root_dir / "sentiment_classifier" / "configs" / "training"/ "config.cfg"
        config_path = str(config_path)
    if training_data_path is None:
        training_data_path = root_dir / "data" / "train.spacy"
        training_data_path = str(training_data_path)
    if testing_data_path is None:
        testing_data_path = root_dir / "data" / "test.spacy"
        testing_data_path = str(testing_data_path)
    if output_path is None:
        output_path = root_dir / "models" 
        output_path = str(output_path)

    train(config_path=config_path, overrides={"paths.train":training_data_path, "paths.dev":testing_data_path}, output_path=output_path)

    spacy_model = load_spacy_model(model_path= Path(output_path) / "model-best")

    example = "the company has no plans to move all production to Russia, although that is where the company is growing"
    doc = spacy_model(example)
    model_signature = infer_signature(model_input = example , model_output = doc.cats)

    experiment = get_or_create_experiment("spacy_classifier", {"project_name":"sentiment_classifier", "framework":"spacy"})

    with mlflow.start_run(run_name="log_spacy_model", experiment_id=experiment.experiment_id) as run:
        mlflow.spacy.log_model(spacy_model = spacy_model, artifact_path = "best_model", signature = model_signature)