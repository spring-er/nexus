"""
config.py — Centralized Configuration for Nexus AI

This module defines ALL application settings in one place using Pydantic's
BaseSettings. It automatically reads values from the .env file and validates
their types.

WHY PYDANTIC SETTINGS?
- Instead of calling os.getenv("SOME_KEY") scattered across 20 files,
  we define every setting here with its type and default value.
- If a required key is missing, Pydantic raises a clear error at startup
  (not buried in a random function call 10 minutes later).
- Autocomplete works in your IDE because settings are typed attributes.

USAGE:
    from config import settings
    print(settings.openrouter_api_key)
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Each attribute corresponds to an environment variable. Pydantic
    automatically converts the variable name to uppercase when looking
    it up. For example, `openrouter_api_key` reads `OPENROUTER_API_KEY`.
    """

    # ── LLM Provider API Keys ──────────────────────────────────────────
    # Optional because not every user will have every provider configured.
    # The app gracefully disables providers whose keys are missing.

    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenRouter (free models available)",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="API key for Groq (free tier)",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="API key for Google AI Studio / Gemini (free tier)",
    )

    # ── Embeddings ─────────────────────────────────────────────────────
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="API key for HuggingFace Inference API",
    )

    # ── Monitoring ─────────────────────────────────────────────────────
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="API key for LangSmith tracing",
    )
    langfuse_public_key: Optional[str] = Field(
        default=None,
        description="Public key for Langfuse analytics",
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for Langfuse analytics",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server URL",
    )

    # ── Ollama (Local) ─────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server",
    )

    # ── Application Settings ───────────────────────────────────────────
    app_name: str = Field(
        default="Nexus AI",
        description="Display name of the application",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode for verbose logging",
    )

    model_config = {
        # Tell Pydantic to read from a .env file in the project root
        "env_file": ".env",
        # If the .env file doesn't exist, don't crash — just use defaults
        "env_file_encoding": "utf-8",
        # Allow extra fields without raising errors (future-proofing)
        "extra": "ignore",
    }


# ── Singleton Instance ─────────────────────────────────────────────────
# We create ONE instance that the entire app imports.
# This way, .env is read exactly once at startup.
settings = Settings()
