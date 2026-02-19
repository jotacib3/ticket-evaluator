"""Application configuration loaded from environment variables.

Uses pydantic-settings to automatically read from a .env file.
All sensitive values (API keys) are stored as SecretStr.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── OpenAI Configuration ───
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"

    # ─── File Paths ───
    input_file: Path = Path("tickets.csv")
    output_file: Path = Path("tickets_evaluated.csv")

    # ─── Evaluation Settings ───
    max_concurrency: int = 3
    max_retries: int = 3
