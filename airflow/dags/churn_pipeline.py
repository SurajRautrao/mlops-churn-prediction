from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging
import sys
from pathlib import Path

# Repo root (airflow/dags/ -> project root); scripts use paths relative to it
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------------------ Default Args ------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}

# ------------------ Task Functions ------------------
def load_data():
    logging.info("Loading and cleaning data...")
    subprocess.run([sys.executable, "-m", "src.data"], cwd=PROJECT_ROOT, check=True)

def train_model():
    logging.info("Training model...")
    subprocess.run([sys.executable, "-m", "src.train"], cwd=PROJECT_ROOT, check=True)

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
