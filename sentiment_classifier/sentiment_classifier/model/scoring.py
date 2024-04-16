import mlflow

from sentiment_classifier.data.retrieval import read_csv
from sentiment_classifier.data.training_dataset import get_train_test_data
from sentiment_classifier.utils.utils import get_root_dir
from sentiment_classifier.utils.utils import read_config


def predict():
    """
    Predict on the test data.
    """
    configs = read_config("train_config")
    root = get_root_dir()
    data_path = root / "data" / "all-data.csv"
    df = read_csv(
        data_path, encoding="latin-1", header=None, names=["sentiment", "text"]
    )

    _, test = get_train_test_data(df)
    registered_model_name = configs["mlflow"]["run"]["registered_model_name"]
    model_uri = f"models:/{registered_model_name}@Champion"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    # Predict on a pandas DataFrame.
    predictions = loaded_model.predict(test)
    print(predictions)
