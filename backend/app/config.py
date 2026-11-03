"""Application configuration, loaded from environment variables / a .env file.

All settings have sensible defaults so the app runs out of the box against a
local Ollama server. Nothing here contains secrets; real values live in .env.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "AI Document Q&A"
    frontend_url: str = "http://localhost:8501"
    data_dir: Path = Field(default=Path("data"))

    # --- Embeddings provider: "ollama" | "sentence_transformers" ---
    embeddings_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    sentence_transformers_model: str = "all-MiniLM-L6-v2"

    # --- LLM provider: "ollama" | "openai" ---
    llm_provider: str = "ollama"
    ollama_chat_model: str = "llama3.1:8b"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"

    # --- Retrieval / chunking ---
    chunk_size: int = 900           # characters per chunk
    chunk_overlap: int = 150        # character overlap between chunks
    top_k: int = 4                  # chunks retrieved per question
    score_threshold: float = 0.25   # min cosine score to treat context as relevant

    # --- Limits ---
    max_upload_mb: int = 25
    request_timeout: int = 120      # seconds for LLM/embedding calls

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
