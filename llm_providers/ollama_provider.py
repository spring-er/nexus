"""
llm_providers/ollama_provider.py — Ollama (Local) LLM Provider

Ollama (ollama.com) runs LLMs locally on your machine. No API key needed,
no internet required, completely free and private. You download models
once, then they run from your hardware.

KEY DIFFERENCE FROM CLOUD PROVIDERS:
    - No API key needed — connects to localhost
    - We check availability by pinging the local Ollama server
    - Users need to have pulled models first (e.g., `ollama pull llama3.1`)
    - Ollama has been OpenAI-compatible since v0.1.14, so we still
      use the openai SDK — same pattern, just pointed at localhost

PREREQUISITES:
    1. Install Ollama: https://ollama.com/download
    2. Start the server: `ollama serve` (or it auto-starts on Mac)
    3. Pull a model: `ollama pull llama3.1`

TRADEOFF VS CLOUD:
    Pros: Free, private, no rate limits, works offline
    Cons: Slower (unless you have a GPU), uses RAM, limited model selection
"""

import httpx
from openai import OpenAI

from config import settings
from llm_providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
)


# ── Common Local Models ────────────────────────────────────────────────
# These are popular models users can pull with `ollama pull <name>`.
# Unlike cloud providers, users must download these before they work.

OLLAMA_COMMON_MODELS = [
    (
        "llama3.1",
        "Llama 3.1 8B (Local)",
        "Meta's latest. Pull with: ollama pull llama3.1",
    ),
    (
        "gemma2",
        "Gemma 2 (Local)",
        "Google's efficient model. Pull with: ollama pull gemma2",
    ),
    (
        "mistral",
        "Mistral 7B (Local)",
        "Fast and capable. Pull with: ollama pull mistral",
    ),
    (
        "qwen2.5",
        "Qwen 2.5 (Local)",
        "Great at code & math. Pull with: ollama pull qwen2.5",
    ),
    (
        "phi3",
        "Phi-3 Mini (Local)",
        "Microsoft's small but mighty model. Pull with: ollama pull phi3",
    ),
]


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider that connects to a local Ollama server.

    Ollama runs on your machine at http://localhost:11434. Since it
    supports the OpenAI-compatible API format, we use the same openai
    SDK — just with base_url pointed at localhost.

    Attributes:
        base_url: The URL where Ollama is running.
        client: An OpenAI SDK client configured for the local Ollama server.
    """

    def __init__(self) -> None:
        """
        Initialize the Ollama provider.

        Points the OpenAI client at the local Ollama server.
        Note: api_key is set to "ollama" — Ollama doesn't need a real key,
        but the OpenAI SDK requires a non-empty string.
        """
        self.base_url = settings.ollama_base_url

        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            # Ollama doesn't need an API key, but the SDK requires one.
            # "ollama" is a conventional placeholder.
            api_key="ollama",
        )

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a conversation to a local Ollama model.

        Same flow as cloud providers: convert messages → call API → wrap response.
        The only difference is the request goes to localhost instead of the internet.
        """
        api_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        content = response.choices[0].message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=content,
            model=model,
            provider="ollama",
            usage=usage,
            raw_response=response,
        )

    def get_available_models(self) -> list[ModelInfo]:
        """
        Return the list of common Ollama models.

        NOTE: These are models the user *could* pull. Whether they're
        actually installed depends on the user's local setup. We list
        common ones as suggestions. In a future enhancement, we could
        query the Ollama API to see which models are actually installed.
        """
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="ollama",
                description=desc,
            )
            for model_id, name, desc in OLLAMA_COMMON_MODELS
        ]

    def is_available(self) -> bool:
        """
        Check if the Ollama server is running locally.

        Unlike cloud providers (which check for an API key), Ollama
        doesn't need a key. Instead, we ping the server to see if
        it's running. If the request fails (connection refused), Ollama
        isn't running.

        We use httpx with a short timeout (2 seconds) so the app
        doesn't hang if Ollama isn't started.
        """
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
