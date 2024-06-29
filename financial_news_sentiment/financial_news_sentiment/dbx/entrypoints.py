from financial_news_sentiment.dbx.job_manager import create_job


def create_and_run_job():
    """
    Create a job in databricks
    """
    name = "test-job"
    description = "test job"
    response = create_job(name, description)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.text)
