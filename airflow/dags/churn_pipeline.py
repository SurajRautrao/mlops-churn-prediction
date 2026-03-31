from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging

# ------------------ Default Args ------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

# ------------------ Task Functions ------------------
def load_data():
    logging.info("📥 Loading and cleaning data...")
    subprocess.run(["python", "src/data.py"], check=True)

def train_model():
    logging.info("🤖 Training model...")
    subprocess.run(["python", "src/train.py"], check=True)

# ------------------ DAG ------------------
with DAG(
    dag_id="churn_training_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    description="Customer Churn Training Pipeline"
) as dag:

    task1 = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    task2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # ------------------ Task Order ------------------
    task1 >> task2
