
# # imagen a usar
# FROM python:3.14-slim

# #Creacion del directorio de la app
# WORKDIR /app

# #Instalacion del gestor de dependencias UV
# RUN pip install --no-cache-dir uv

# #archivo que contiene las depencias del proyecto
# COPY pyproject.toml ./

# ENV PYTHONUNBUFFERED=1 \
#     UV_PROJECT_ENVIRONMENT=/opt/venv \
#     PATH="/opt/venv/bin:$PATH"

# RUN uv sync --no-dev

# #Archivos a copiar
# COPY 01-Orchestration/ ./01-Orchestration/
# COPY src/ ./src/

# EXPOSE 5000 
# EXPOSE 4200

# CMD ["python", "/app/01-Orchestration/Prefect-pipeline/pipeline.py"]
FROM python:3.14-slim
 
RUN pip install --no-cache-dir uv
 
ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
 
WORKDIR /app
 
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
 
 
COPY 01-Orchestration/ ./01-Orchestration/
COPY 02-Deployment/ ./02-Deployment/
 
WORKDIR /workspace
 
 
EXPOSE 4200 5000 8000
 
CMD ["python", "/app/01-Orchestration/Prefect-pipeline/pipeline.py"]
 
 