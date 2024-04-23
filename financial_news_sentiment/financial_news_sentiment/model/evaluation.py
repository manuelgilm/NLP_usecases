from typing import Dict
from typing import List

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def classification_metrics(
    y_true: List[str], y_pred: List[str], prefix=""
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :param prefix: Prefix for metric names.
    :return: Dictionary of classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
