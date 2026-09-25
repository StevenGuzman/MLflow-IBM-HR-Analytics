"""
API sencilla para predecir la rotacion (Attrition) de un empleado.

Carga desde el MLflow Model Registry la version con alias "champion" de
IBM-Attrition-Predictor (el XGBoost) junto con su preprocessor (el
DictVectorizer que se entreno en el mismo run), y expone un endpoint
POST /predict que recibe los datos de una persona.

Ademas guarda empleados en un SQLite (CRUD en /employees) con su prediccion,
y sirve en / una pagina sencilla para administrarlos.

Ejecucion local (con el tracking server en localhost:5000):
    MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn app:app --reload
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlflow
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "IBM-Attrition-Predictor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
# El modelo se entreno como regresion (reg:squarederror) sobre un target 0/1,
# asi que su salida es un score continuo que interpretamos como probabilidad.
THRESHOLD = float(os.getenv("ATTRITION_THRESHOLD", "0.5"))
DB_PATH = os.getenv("DB_PATH", "employees.db")
STATIC_DIR = Path(__file__).parent / "static"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

app = FastAPI(
    title="IBM HR Attrition API",
    description="Predice la probabilidad de rotacion de un empleado con el modelo champion de MLflow.",
)

# Se llena la primera vez que se pide una prediccion, asi la API puede
# arrancar aunque el pipeline de entrenamiento todavia no haya terminado.
_model = {}


def load_model():
    """Carga el booster y el preprocessor de la version champion."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    # train_model guarda el model_id del DictVectorizer como tag del run
    run = client.get_run(version.run_id)
    preprocessor_model_id = run.data.tags["preprocessor_model_id"]

    _model["booster"] = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    _model["dv"] = mlflow.sklearn.load_model(f"models:/{preprocessor_model_id}")
    _model["version"] = version.version


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data TEXT NOT NULL,
            attrition_probability REAL NOT NULL,
            attrition TEXT NOT NULL,
            model_version TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


class Employee(BaseModel):
    """Mismas columnas que usa el pipeline (sin Attrition ni las que elimina validate_data)."""

    Age: int = Field(ge=18, le=70)
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    DailyRate: int
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    DistanceFromHome: int
    Education: int = Field(ge=1, le=5)
    EducationField: Literal[
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
    ]
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    Gender: Literal["Female", "Male"]
    HourlyRate: int
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources",
    ]
    JobSatisfaction: int = Field(ge=1, le=4)
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int
    PerformanceRating: int = Field(ge=1, le=4)
    RelationshipSatisfaction: int = Field(ge=1, le=4)
    StockOptionLevel: int = Field(ge=0, le=3)
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int = Field(ge=1, le=4)
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 41, "BusinessTravel": "Travel_Rarely", "DailyRate": 1102,
                "Department": "Sales", "DistanceFromHome": 1, "Education": 2,
                "EducationField": "Life Sciences", "EnvironmentSatisfaction": 2,
                "Gender": "Female", "HourlyRate": 94, "JobInvolvement": 3,
                "JobLevel": 2, "JobRole": "Sales Executive", "JobSatisfaction": 4,
                "MaritalStatus": "Single", "MonthlyIncome": 5993, "MonthlyRate": 19479,
                "NumCompaniesWorked": 8, "OverTime": "Yes", "PercentSalaryHike": 11,
                "PerformanceRating": 3, "RelationshipSatisfaction": 1,
                "StockOptionLevel": 0, "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 0, "WorkLifeBalance": 1, "YearsAtCompany": 6,
                "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 5,
            }
        }
    }


class Prediction(BaseModel):
    attrition_probability: float
    attrition: Literal["Yes", "No"]
    threshold: float
    model_version: str


class EmployeeIn(Employee):
    Name: str = Field(min_length=1, max_length=100)


class EmployeeRecord(Prediction):
    id: int
    Name: str
    data: Employee
    updated_at: str


def _to_record(row: sqlite3.Row) -> EmployeeRecord:
    return EmployeeRecord(
        id=row["id"],
        Name=row["name"],
        data=Employee(**json.loads(row["data"])),
        attrition_probability=row["attrition_probability"],
        attrition=row["attrition"],
        threshold=THRESHOLD,
        model_version=row["model_version"],
        updated_at=row["updated_at"],
    )


def _save(employee: EmployeeIn, employee_id: int | None = None) -> EmployeeRecord:
    """Calcula la prediccion y guarda (o actualiza) el empleado."""
    features = Employee(**employee.model_dump(exclude={"Name"}))
    prediction = predict(features)
    values = (
        employee.Name,
        features.model_dump_json(),
        prediction.attrition_probability,
        prediction.attrition,
        prediction.model_version,
        datetime.now().isoformat(timespec="seconds"),
    )
    with get_db() as conn:
        if employee_id is None:
            cursor = conn.execute(
                "INSERT INTO employees (name, data, attrition_probability, attrition,"
                " model_version, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                values,
            )
            employee_id = cursor.lastrowid
        else:
            cursor = conn.execute(
                "UPDATE employees SET name = ?, data = ?, attrition_probability = ?,"
                " attrition = ?, model_version = ?, updated_at = ? WHERE id = ?",
                (*values, employee_id),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Empleado no encontrado")
        row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    return _to_record(row)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_model)}


@app.post("/predict", response_model=Prediction)
def predict(employee: Employee):
    if not _model:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo cargar {MODEL_NAME}@{MODEL_ALIAS} desde MLflow: {e}",
            )

    # Mismo preprocesamiento que create_features: dict -> DictVectorizer
    X = _model["dv"].transform([employee.model_dump()])
    score = float(_model["booster"].predict(xgb.DMatrix(X))[0])
    probability = min(max(score, 0.0), 1.0)

    return Prediction(
        attrition_probability=round(probability, 4),
        attrition="Yes" if probability >= THRESHOLD else "No",
        threshold=THRESHOLD,
        model_version=str(_model["version"]),
    )


@app.get("/employees", response_model=list[EmployeeRecord])
def list_employees():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY id DESC").fetchall()
    return [_to_record(row) for row in rows]


@app.get("/employees/{employee_id}", response_model=EmployeeRecord)
def get_employee(employee_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Empleado no encontrado")
    return _to_record(row)


@app.post("/employees", response_model=EmployeeRecord, status_code=201)
def create_employee(employee: EmployeeIn):
    return _save(employee)


@app.put("/employees/{employee_id}", response_model=EmployeeRecord)
def update_employee(employee_id: int, employee: EmployeeIn):
    return _save(employee, employee_id)


@app.delete("/employees/{employee_id}", status_code=204)
def delete_employee(employee_id: int):
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Empleado no encontrado")
