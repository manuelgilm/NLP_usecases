from typing import Dict
from typing import List

from transformers import pipeline

from financial_news_sentiment.utils.utils import read_config


class ShotClassifier(object):
    """
    ShotClassifier class for sentiment analysis.
    """

    def __init__(self):
        """
        Initialize the ShotClassifier object.
        """
        self.config = read_config("pipelines/shot_classifier")
        self.labels = self.config["labels"]
        self.task = self.config["model"]["task"]
        self.checkpoint = self.config["model"]["checkpoint"]
        self.pipeline = self.__get_transformer_pipeline()

    def predict(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Get sentiment analysis predictions for a list of texts.

        :param texts: List of texts.
        :return: List of sentiment analysis predictions.
        """
        return self.__get_prediction(texts)

    def __get_transformer_pipeline(self) -> "pipeline":
        """
        Get the transformer pipeline.
        """
        classifier = pipeline(task=self.task, model=self.checkpoint)
        return classifier

    def __get_prediction(self, texts: List[str]):
        """
        Get sentiment analysis predictions for a list of texts.

        :param texts: List of texts.
        :return: List of sentiment analysis predictions.
        """
        predictions = self.pipeline(texts, candidate_labels=self.labels)
        return self.__process_output(predictions)

    def __process_output(
        self, predictions: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Process the output.

        :param predictions: List of predictions.
        :return: Processed predictions.
        """
        processed_prediction = [
            {
                "text": prediction["sequence"],
                "label": prediction["labels"][0],
                "score": prediction["scores"][0],
            }
            for prediction in predictions
        ]
        return processed_prediction
