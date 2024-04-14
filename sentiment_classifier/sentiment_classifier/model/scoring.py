import mlflow

from sentiment_classifier.data.retrieval import read_csv
from sentiment_classifier.data.training_dataset import get_train_test_data
from sentiment_classifier.utils.utils import get_root_dir


def predict():
    """
    Predict on the test data.
    """
    root = get_root_dir()
    data_path = root / "data" / "all-data.csv"
    df = read_csv(
        data_path, encoding="latin-1", header=None, names=["sentiment", "text"]
    )

    _, test = get_train_test_data(df)
    runs = mlflow.search_runs(experiment_names=["spacy_classifier"])
    run_id = runs["run_id"].values[0]
    model_uri = f"runs:/{run_id}/best_model"
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    # Predict on a pandas DataFrame.
    predictions = loaded_model.predict(test)
    print(predictions)
