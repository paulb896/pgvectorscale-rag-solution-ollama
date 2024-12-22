import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OllamaSettings(LLMSettings):
    """Ollama-specific settings extending LLMSettings."""
    default_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1") # Used for chat queries
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large") # Used for generating embeddings
    temperature: float = os.getenv("OLLAMA_TEMPERATURE", 0.7)
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL", "postgres://postgres:password@127.0.0.1:5432/postgres"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""
    table_name: str = os.getenv("VECTOR_DATABASE_EMBEDDING_TABLE_NAME", "embeddings")
    embedding_dimensions: int = os.getenv("VECTOR_DATABASE_EMBEDDING_DIMENSIONS", 1024)
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
