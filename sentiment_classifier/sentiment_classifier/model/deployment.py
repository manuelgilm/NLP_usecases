import argparse
import json

import requests

PORT = 5000

parser = argparse.ArgumentParser(description="Scoring Data")
parser.add_argument(
    "--model_input",
    help="The input to the model",
)


def online_inference():
    """
    Send a request to the model.
    """

    args = parser.parse_args()
    model_input = args.model_input

    if model_input:
        payload = json.dumps(
            {"dataframe_split": {"columns": ["text"], "data": [model_input]}}
        )
    else:
        raise Exception("Model input is required")

    try:
        response = requests.post(
            url=f"http://localhost:{PORT}/invocations",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")
