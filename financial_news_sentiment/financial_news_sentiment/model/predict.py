import pandas as pd

from financial_news_sentiment.data.retrieval import get_train_test_data
from financial_news_sentiment.data.retrieval import read_data
from financial_news_sentiment.model.evaluation import classification_metrics
from financial_news_sentiment.model.evaluation import track_run
from financial_news_sentiment.model.pipelines import ShotClassifier
from financial_news_sentiment.utils.utils import add_encoding
from financial_news_sentiment.utils.utils import as_dataframe
from financial_news_sentiment.utils.utils import get_root_path
from financial_news_sentiment.utils.utils import read_config


def predict():
    """
    Get sentiment analysis prediction for a single text.

    :param text: Text.
    :return: Sentiment analysis prediction.
    """

    df = read_data()
    # transform the data to a binary classification problem
    df["sentiment"] = df["sentiment"].apply(
        lambda x: "positive" if x == "positive" else "negative"
    )
    print(df["sentiment"].value_counts())

    _, test_df = get_train_test_data(df)
    classifier = ShotClassifier()
    predictions = classifier.predict(test_df["text"].to_list())
    predictions_df = as_dataframe(predictions)
    predictions_df["sentiment"] = test_df["sentiment"].to_list()

    root = get_root_path()
    output_path = (
        root
        / "financial_news_sentiment"
        / "output"
        / "shot_classifier_test_dataset_predictions.csv"
    )
    output_path = add_encoding(output_path)
    predictions_df.to_csv(output_path, index=False)
    print(predictions_df.head())


def track_experiment(experiment_config: str):
    """
    Track the experiment in MLflow.
    """
    config = read_config(experiment_config)

    experiment_name = config["experiment_name"]
    experiment_tags = config["experiment_tags"]
    run_name = config["run_name"]
    run_tags = config["run_tags"]
    data_path = config["dataset_path"]
    df = pd.read_csv(data_path)
    metrics = classification_metrics(
        df["label"], df["sentiment"], prefix="test"
    )
    track_run(
        experiment_name, experiment_tags, run_name, run_tags, metrics=metrics
    )


def log_experiments():
    """
    Log experiments in MLflow.
    """
    track_experiment("experiments/shot_classifier")
