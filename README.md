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

# AUTOR(ES) 
* Kelly Zenith Panizza Ordoñez
* Laura Marcela Hoyos Perez
* Steven Guzman Angulo
* Zamir Flores Cordoba