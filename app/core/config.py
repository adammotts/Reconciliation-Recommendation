# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Reconciliation"
    API_V1_STR: str = "/api/v1"

settings = Settings()
