from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.serve.batch_score_step9 import score_listings


with DAG(
    dag_id="trustlens_batch_score",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["trustlens", "fraud", "batch"],
) as dag:
    run_batch_score = PythonOperator(
        task_id="run_batch_score",
        python_callable=score_listings,
    )

    run_batch_score
