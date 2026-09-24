
"""
Data splitting tasks.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import task

from ..config import TARGET_COLUMN



@task(
    name="split_data",
    description="Split dataset into training and validation sets"
)
def split_data(df: pd.DataFrame):
    return train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )