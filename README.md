# IBM-HR-Analytics-Predictions

## DESCRIPCION DEL PROYECTO
El presente proyecto consiste en la creacion, desarrollo, seguimiento, orquestación y despliegue de modelos de **Machine Learning** a traves de la creacion de un Pipeline, que permiten determinar la calidad de los modelos elaborados, a traves de servicios desplegados en una imagen dockerizada.

## ESTRUCTURA DEL PROYECTO
```
MLflow-IBM-HR-Analytics
├── 01-Orchestration/           # Aplicacion creada con streamlit
|   |── src/                    # Datos limpios y transformados
|       |── config/             # archivos de configuracion para el pipeline y MLFlow
|       |   |── __init__.py     # variables para la configuracion del pipeline
|       |   |── constants.py    # valores constantes para el pipeline
|       |   |── mlflow_setup.py # configuracion de la herramienta MLflow
|       |── data                # archivos que contienen la Data 
|       |   |── __init__.py     # importacion de modulos 
|       |   |── loaders.py      # tareas para leer Data
|       |   |── validators.py   # validaciones en la Data
|       |── feature             # funcionalidades de ingenieria
|       |   |── __init__.py     # importacion de modulos
|       |   |── engineering.py  # ingenieria aplicada a la Data
|       |── models              # modelos creados
|       |   |── __init__.py     # importacion de modulos
|       |   |── model_registry.py # tracking de modelos creados
|       |   |── optimization.py # optimizacion de hiperparámetros
|   |── pipeline.py             # archivo que lógica principal del pipeline del proyecto
|
├── 02-Deployment/
|   |── web-service/
|       |── app.py              # API FastAPI que sirve el modelo champion
|       |── static/index.html   # página CRUD de empleados con su predicción
|
├── src/            
|   |── __init__.py             # archivo inicializador 
├── python-version              # version de python utilizada
├── dockerfile                  # imagen creada para despliegue
└── pyproject.toml              # vibrerias, dependencias empleadas en el proyecto
```



## Tecnologias utilizadas
- ipykernel>=7.3.0
- kagglehub>=1.0.2
- mlflow>=3.16.1
- openai>=3.16.2
- optuna>=5.0.0
- pandas>=3.0.6
- prefect>=3.8.6
- scikit-learn>=1.9.1
- xgboost>=3.4.1



## Instalacion
Siga estos pasos de forma secuencial para preparar su entorno local y asegurar que el modelo funcione correctamente:

### 1. Clonar el repositorio
Primero, obtenga una copia local del proyecto ejecutando el siguiente comando en su terminal:
```bash
git clone (https://github.com/StevenGuzman/MLflow-IBM-HR-Analytics.git)
cd MLflow-IBM-HR-Analytics
```
### 2. Crear el entorno virtual
Es fundamental aislar las librerías del proyecto para evitar conflictos de versiones. Puede realizarlo de dos formas:
*   **Con Python estándar:** `python -m venv .venv`
*   **Con uv (Recomendado):** `uv venv`


### 3. Sincronización de librerías y selección de Kernel
Una vez creado el entorno, debe instalar las dependencias y vincular el editor:

**Sincronizar librerías:**
*   Si utiliza **uv**, ejecute el siguiente comando para instalar automáticamente lo definido en el `pyproject.toml`
 ```
uv sync
 ```


### 4. Construccion de la imagen y contenedores
 Desde la raíz del proyecto, donde se encuentra docker-compose.yml:
```
docker compose up --build -d
```

verificar que la imagen sea creada de manera satisfactoria
```
docker compose ps
```

Acceder a la aplicación a traves de los puertos definidos en los contenedores, ejemplo:
```
http://localhost:5000

```

### 5. API de predicción (FastAPI)
El contenedor `ibm-hr-api` carga desde MLflow la versión con alias `champion` de
`IBM-Attrition-Predictor` y expone un endpoint para predecir la rotación de un empleado.
La primera predicción falla con `503` si el pipeline todavía no ha registrado un modelo.

Página web para registrar, editar y eliminar empleados y ver su probabilidad de rotación
(los empleados se guardan en SQLite, en el volumen `api-data`):
```
http://localhost:8000
```

Documentación interactiva (Swagger), con un ejemplo listo para probar:
```
http://localhost:8000/docs
```

Ejemplo con curl:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"Age":41,"BusinessTravel":"Travel_Rarely","DailyRate":1102,"Department":"Sales","DistanceFromHome":1,"Education":2,"EducationField":"Life Sciences","EnvironmentSatisfaction":2,"Gender":"Female","HourlyRate":94,"JobInvolvement":3,"JobLevel":2,"JobRole":"Sales Executive","JobSatisfaction":4,"MaritalStatus":"Single","MonthlyIncome":5993,"MonthlyRate":19479,"NumCompaniesWorked":8,"OverTime":"Yes","PercentSalaryHike":11,"PerformanceRating":3,"RelationshipSatisfaction":1,"StockOptionLevel":0,"TotalWorkingYears":8,"TrainingTimesLastYear":0,"WorkLifeBalance":1,"YearsAtCompany":6,"YearsInCurrentRole":4,"YearsSinceLastPromotion":0,"YearsWithCurrManager":5}'
```

Respuesta:
```
{"attrition_probability": 0.572, "attrition": "Yes", "threshold": 0.5, "model_version": "2"}
```
`attrition_probability` es el score del modelo recortado a [0, 1]; si supera el umbral
(`ATTRITION_THRESHOLD`, 0.5 por defecto) la predicción es `"Yes"`.

# AUTOR(ES) 
* Kelly Zenith Panizza Ordoñez
* Laura Marcela Hoyos Perez
* Steven Guzman Angulo
* Zamir Flores Cordoba