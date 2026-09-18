"""
Data loading tasks with caching and retry logic.
"""

import kagglehub
import pandas as pd
from prefect import task
from prefect.logging import get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    name="load_data",
    description="Download IBM HR Analytics Employee Attrition dataset from Kaggle",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
    persist_result=True,
)
def read_dataframe() -> pd.DataFrame:
    """
    Load IBM HR Analytics Employee Attrition & Performance dataset from parquet file with caching and retry logic.
    
    Returns:
        DataFrame with employees data
    """
    logger = get_run_logger()

    #logger.info("Downloading IBM HR Analytics Attrition dataset")

    # Download latest version
    
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")

    logger.info(f"Dataset downloaded to: {path}")

    # Leer el archivo CSV
    file_path = f"{path}/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    logger.info(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)

    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    return df
