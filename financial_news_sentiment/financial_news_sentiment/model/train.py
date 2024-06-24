from functools import partial
from typing import Any
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from financial_news_sentiment.data.retrieval import get_train_test_data
from financial_news_sentiment.data.retrieval import read_data
from financial_news_sentiment.utils.utils import read_config


def get_tokenizer():
    """
    Get the tokenizer.

    :return: Tokenizer.
    """
    config = read_config("train")
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def tokenize_text(examples: Dict[str, Any], tokenizer):
    """
    Tokenize the text.

    :param text: Text.
    :param tokenizer: Tokenizer.
    :return: Tokenized text.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def get_model():
    """
    Get the model.

    :return: Model.
    """
    config = read_config("train")
    model = config["model"]
    labels = config["labels"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=len(labels), ignore_mismatched_sizes=True
    )
    return model


def get_trainer_arguments():
    """
    Get the trainer arguments.

    :return: Trainer arguments.
    """
    config = read_config("train")

    return TrainingArguments(
        output_dir=config["output_dir"],
        label_names=config["labels"],
        evaluation_strategy=config["evaluation_strategy"],
    )


def compute_metrics(eval_pred, metric):
    """
    Compute metrics.

    :param eval_pred: Evaluation predictions.
    :return: Metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def process_dataset(df: pd.DataFrame):
    """
    Process the dataset.

    :param split: Split.
    :param df: Dataframe.
    :return: Dataset.
    """
    dataset = Dataset.from_pandas(df)
    return dataset


def get_tokenized_data(ds):
    """
    Get the tokenized data.

    :param df: Dataframe.
    :return: Tokenized data.
    """
    tokenizer = get_tokenizer()
    tokenized_data = ds.map(
        partial(tokenize_text, tokenizer=tokenizer), batched=True
    )
    return tokenized_data


def get_trainer():
    """
    Train the model.
    """

    df = read_data()

    # transform the data to a binary classification problem
    df["sentiment"] = df["sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    train_df, test_df = get_train_test_data(df)

    # change column name from sentiment to "label"
    train_df = train_df.rename(columns={"sentiment": "label"})
    test_df = test_df.rename(columns={"sentiment": "label"})

    # get only a small subset of the data
    train_df = train_df.head(100)
    test_df = test_df.head(100)

    train_ds = process_dataset(train_df)
    test_ds = process_dataset(test_df)

    tokenized_train_dataset = get_tokenized_data(train_ds)
    tokenized_test_dataset = get_tokenized_data(test_ds)

    model = get_model()
    trainer_arguments = get_trainer_arguments()

    metric = evaluate.load("f1")

    trainer = Trainer(
        model=model,
        args=trainer_arguments,
        compute_metrics=partial(compute_metrics, metric=metric),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )
    trainer.train()
