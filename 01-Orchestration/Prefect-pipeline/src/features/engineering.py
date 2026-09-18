"""
Feature engineering tasks.
"""

from typing import Tuple, Optional

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from prefect import task, get_run_logger
from prefect.artifacts import create_table_artifact

from ..config import CATEGORICAL_FEATURES


@task(name="create_features", description="Create feature matrix using DictVectorizer")
# Devuelve X: matriz de variables para el modelo.
# dv: el vectorizador utilizado.
def create_features(df: pd.DataFrame, dv: Optional[DictVectorizer] = None) -> Tuple[any, DictVectorizer]:
    """
    Create feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        dv: Pre-fitted DictVectorizer (optional)

    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """
    logger = get_run_logger()
    
    # Crear feature PU_DO combinado
    df_features = df.copy()
    df_features['PU_DO'] = df_features['PULocationID'].astype(str) + '_' + df_features['DOLocationID'].astype(str)
    
    # Usar solo PU_DO y trip_distance
    #Convierte las columnas en listas de diccionarios para preparar los datos para DictVectorizer.
    dicts = df_features[['PU_DO', 'trip_distance']].to_dict(orient='records')
    
    logger.info(f"Created {len(dicts)} feature dictionaries with PU_DO combined")
   #DictVectorizer convierte variables categóricas en variables numéricas mediante one-hot encoding.
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        
        # Create artifact with feature information
        feature_info = [
            ["Total Features", X.shape[1]],
            ["Feature Names Sample", ", ".join(dv.feature_names_[:5]) + "..."],
            ["Sparse Matrix Shape", f"{X.shape[0]} x {X.shape[1]}"]
        ]
        #Se esta creando un tabla artifact de Prefect con información sobre las variables.
        create_table_artifact(
            key="feature-info",
            table=feature_info,
            description="Feature matrix information"
        )
    else:
        X = dv.transform(dicts)

    return X, dv
