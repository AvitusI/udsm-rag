from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseConfig(BaseSettings):
    PINECONE_API_KEY_SECRET: Optional[str]
    AWS_REGION: Optional[str]
    AWS_ACCESS_KEY_ID: Optional[str]
    AWS_SECRET_ACCESS_KEY: Optional[str]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )