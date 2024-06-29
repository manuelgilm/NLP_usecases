import json
import os
from pprint import pprint
from typing import Any
from typing import Dict

import requests
from dotenv import load_dotenv

from financial_news_sentiment.utils.utils import read_config

load_dotenv()

TOKEN = os.environ.get("TOKEN", None)
HOST = os.environ.get("HOST", None)

if TOKEN is None or HOST is None:
    raise ValueError(
        "Please provide the databricks token and host in the .env file"
    )

BASE_URL = f"{HOST}/api/2.1/jobs/"

# headers with authentication token and content type
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def create_job(name: str, description: str):
    """
    Create a job in databricks

    :param name: str: name of the job
    :param description: str: description of the job
    :param cluster_name: str: name of the cluster to run the job
    :return: requests.models.Response: response from the databricks API
    """
    cluster_config = get_cluster_configuration()
    tasks = read_config("tasks/fine-tuning")
    endpoint = BASE_URL + "create"

    tasks[0].update(
        {
            "new_cluster": cluster_config,
        },
    )
    payload = {
        "name": name,
        "description": description,
        "tasks": tasks,
    }
    pprint(payload)
    response = requests.post(
        endpoint, data=json.dumps(payload), headers=HEADERS
    )
    return response


def get_cluster_configuration() -> Dict[str, Any]:
    """
    Get the configuration for a cluster in databricks

    :param cluster_name: str: name of the cluster
    :return: dict: configuration for the cluster
    """
    # provide configuration for a basic cluster in databricks with ML runtime
    return {
        "spark_version": "11.3.x-cpu-ml-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "num_workers": 2,
    }
