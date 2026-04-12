"""Settings module: pydantic-settings configuration loaded from .env and environment variables.

Input: environment variables or a .env file in the working directory.
Output: a singleton Settings instance importable as `from genai_batch_ie_rfs.settings import settings`.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment or a .env file.

    All fields can be overridden by environment variables using the same name
    (case-insensitive). A .env file in the working directory is loaded automatically.

    Attributes:
        openai_api_key: OpenAI API key. Required for the OpenAI backend.
        anthropic_api_key: Anthropic API key. Required for the Anthropic backend.
        default_model: Model identifier used when no model is specified explicitly.
        default_backend: Backend name to use by default. Either "openai" or "anthropic".
        openai_base_url: Optional custom base URL for the OpenAI client (e.g., OpenRouter).
        max_workers: Default number of concurrent threads for classify_df.
        chunk_size: Default number of rows per LLM request chunk.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key.")
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key.")
    default_model: str = Field(default="gpt-4.1-mini", description="Default LLM model name.")
    default_backend: str = Field(default="openai", description="Default backend: openai or anthropic.")
    openai_base_url: str | None = Field(default=None, description="Custom OpenAI base URL.")
    max_workers: int = Field(default=20, description="Default concurrent worker count.")
    chunk_size: int = Field(default=5, description="Default rows per LLM chunk.")


settings = Settings()
