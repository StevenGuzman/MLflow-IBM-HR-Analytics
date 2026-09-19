"""
Data validation tasks.
"""
import pandas as pd
from prefect import task, get_run_logger

from ..config import MIN_RECORDS, NULL_THRESHOLD


@task(name="validate_data", description="Validate data quality before processing")
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data quality with basic checks.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Validated DataFrame
    """
    logger = get_run_logger()
    
    # Check for nulls
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    
    # Basic validation - solo warnings, no errores
    if len(df) < MIN_RECORDS: #cuenta cuantas filas tiene el dataframe y si es menor a MIN RECORDS, lanza un warning
        logger.warning(f"Low data volume: {len(df)} rows (recommended: {MIN_RECORDS})")
    
    if null_pct > NULL_THRESHOLD: #si el porcentaje de nulos es mayor a NULL_THRESHOLD, lanza un warning
        logger.warning(f"High null percentage: {null_pct:.2f}%")

    #Quitando columnas que no aportan al modelo o analisis.
    df = df.drop(columns=['EmployeeCount','StandardHours','Over18','EmployeeNumber',])
   
    logger.info(f"Data loaded: {len(df)} rows, {null_pct:.2f}% nulls") #Mensaje para saber cuantas filas se cargaron y el porcentaje de nulos

    return df
