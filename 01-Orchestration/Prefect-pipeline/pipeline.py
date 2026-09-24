#!/usr/bin/env python
"""
IBM HR Analytics Employee Attrition & Performance Pipeline - Modular Architecture
Main orchestration flow using Prefect with domain-driven design.
"""

import logging
import mlflow
from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.config.mlflow_setup import setup_mlflow
from src.config.constants import (TARGET_COLUMN,MLFLOW_EXPERIMENT_NAME,MLFLOW_UI_URL)
from src.data.loaders import read_dataframe
from src.data.validators import validate_data
from src.data.split_data import split_data
from src.features.engineering import create_features
from src.models.optimization import optimize_hyperparameters,train_model
from src.models.model_registry import register_best_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
setup_mlflow()


@flow(
    name="IBM HR Analytics Employee Attrition Performance Pipeline",
    description="End-to-end ML pipeline for employee attrition prediction with Optuna optimization",
    log_prints=True
)
def attrition_prediction_flow() -> str:
    """
    Main pipeline flow for Attrition prediction.

    Returns:
        MLflow run ID
    """
    logger = get_run_logger()

    # Load training data
    df = read_dataframe()
    
    # Validate training data
    df = validate_data(df)

    # Split data into training and validation sets
    df_train, df_val = split_data(df)

    # Create features
    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)

    # Prepare targets
    #y_train = df_train[TARGET_COLUMN].values
    #y_val = df_val[TARGET_COLUMN].values
    y_train = df_train[TARGET_COLUMN].map({
    'No': 0,
    'Yes': 1
      }).values

    y_val = df_val[TARGET_COLUMN].map({
    'No': 0,
    'Yes': 1
     }).values

    # Optimize hyperparameters with Optuna
    logger.info("Starting hyperparameter optimization...")
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Train model with optimized parameters
    logger.info("Training final model with optimized parameters...")
    model_run_id, rmse = train_model(X_train, y_train, X_val, y_val, dv, best_params)

    # Register best model in MLflow Model Registry
    logger.info("Registering best model in MLflow Model Registry...")
    model_version = register_best_model(
        run_id=model_run_id,
        rmse=rmse,
        model_name="IBM-Attrition-Predictor"
    )
    logger.info(f"Model registered as version {model_version}")

    # Create final pipeline artifact with enhanced information
    pipeline_summary = f"""
    # Pipeline Execution Summary

    ## Data
    - **Dataset**: IBM HR Analytics Employee Attrition
    - **Records**: 1470
    - **Features**: 51
    - **Training Samples**: {len(y_train):,}
    - **Validation Samples**: {len(y_val):,}
    - **Features**: {X_train.shape[1]:,}

    ## Results
    - **RMSE**: {rmse:.4f}
    - **MLflow Run ID**: [{model_run_id}]({MLFLOW_UI_URL})
    - **MLflow Experiment**: {MLFLOW_EXPERIMENT_NAME}
    - **Registered Model**: IBM-Attrition-Predictor v{model_version}

    ## Next Steps
    1. [Review model in MLflow Model Registry]({MLFLOW_UI_URL}/#/models/IBM-Attrition-Predictor)
    2. La version se registra con el alias "candidate"; solo se promueve a "champion"
       si mejora el RMSE del champion actual (ver register_best_model)
    3. Use deployment module to serve the registered model
    4. Compare with previous model versions

    ## Quick Links
    - [Prefect Cloud Dashboard](https://app.prefect.cloud)
    - [MLflow Tracking UI]({MLFLOW_UI_URL})
    """

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=pipeline_summary,
        description="Complete pipeline execution summary"
    )

    return model_run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict leave or quit employees using Prefect.')
    #parser.add_argument('--year', type=int, default=None, help='Year of the data to train on (default: se detecta automáticamente el periodo más reciente disponible)')
    #parser.add_argument('--month', type=int, default=None, help='Month of the data to train on (default: se detecta automáticamente el periodo más reciente disponible)')
    args = parser.parse_args()

    try:
        # Run the flow
        model_run_id = attrition_prediction_flow()
        print("\nPipeline completed successfully!")
        print(f"MLflow run_id: {model_run_id}")
        print(f"View results at: {mlflow.get_tracking_uri()}")
        print(f"Model registered in MLflow Model Registry: IBM-Attrition-Predictor")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(model_run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


 # uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                                                 