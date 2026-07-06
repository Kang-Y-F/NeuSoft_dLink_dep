from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DB: str

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    DASHSCOPE_API_KEY: str

    JAVA_LAB_API_BASE_URL: str
    SERVICE_BASE_URL: str = "http://localhost:8000"

    OSS_BUCKET: str
    OSS_ENDPOINT: str
    OSS_ACCESS_KEY_ID: str
    OSS_ACCESS_KEY_SECRET: str
    OSS_BASE_URL: str

    PGVECTOR_HOST: str
    PGVECTOR_PORT: int
    PGVECTOR_USER: str
    PGVECTOR_PASSWORD: str
    PGVECTOR_DB : str

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()