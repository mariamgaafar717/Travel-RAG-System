"""Application configuration and settings."""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_TITLE: str = "Travel RAG System API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Retrieval-Augmented Generation API for Egyptian tourism destinations"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    RELOAD: bool = os.getenv("RELOAD", "False").lower() == "true"
    
    # Model con
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 100
    CHUNK_OVERLAP: int = 20
    TOP_K_RESULTS: int = 5
    
    # Data paths
    DATA_DIR: Path = Path(__file__).parent.parent.parent / "data"
    EMBEDDINGS_PATH: Path = DATA_DIR / "embeddings.pkl"
    FAISS_INDEX_PATH: Path = DATA_DIR / "faiss_index.bin"
    CHUNKS_PATH: Path = DATA_DIR / "chunks.pkl"
    DATASET_PATH: Path = DATA_DIR / "egypt_places.json"
    
    # LLM con
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    USE_GPU: bool = os.getenv("USE_GPU", "True").lower() == "true"
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour in seconds
    ENABLE_CACHE: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
