"""
Feature engineering tasks.
"""

from typing import Tuple, Optional

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from prefect import task, get_run_logger


@task(
    name="create_features",
    description="Create feature matrix using DictVectorizer"
)
def create_features(
    df: pd.DataFrame,
    dv: Optional[DictVectorizer] = None
) -> Tuple[any, DictVectorizer]:

    """
    Create feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        dv: Pre-fitted DictVectorizer (optional)

    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """

    logger = get_run_logger()

    
    # 1. Crear una copia para no modificar el DataFrame original
    df_features = df.copy()

    # 2. Convertir la variable objetivo Attrition
    #    No se incluye en las features
    if "Attrition" in df_features.columns:
        df_features = df_features.drop(columns=["Attrition"])

    # 3. Convertir variables categóricas a string

    columnas_categoricas = [
        "BusinessTravel",
        "Department",
        "Gender",
        "MaritalStatus",
        "EducationField",
        "JobRole",
        "OverTime"
    ]

    for col in columnas_categoricas:
        if col in df_features.columns:
            df_features[col] = df_features[col].astype(str)

    # 4. Convertir el DataFrame a diccionarios

    dicts = df_features.to_dict(orient="records")

    logger.info(
        f"Created {len(dicts)} feature dictionaries"
    )

    # 5. Crear y entrenar DictVectorizer

    if dv is None:

        dv = DictVectorizer()

        X = dv.fit_transform(dicts)

        logger.info(
            f"DictVectorizer fitted with {X.shape[1]} features"
        )

    # 6. Utilizar DictVectorizer ya entrenado

    else:

        X = dv.transform(dicts)

        logger.info(
            f"Transformed data using existing DictVectorizer"
        )

    return X, dv