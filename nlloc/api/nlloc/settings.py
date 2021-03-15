from pydantic import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name = 'NLLOC_API'
    project_root: Optional[str] = None
    root: Optional[str] = None

    class Config:
        env_prefix: str = "NLLOC_"


settings = Settings()