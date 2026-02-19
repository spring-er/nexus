"""
llm_providers/groq_provider.py — Groq LLM Provider

Groq (groq.com) runs LLMs on custom LPU (Language Processing Unit) chips,
making inference extremely fast. Their free tier offers several models
with generous rate limits.

Like OpenRouter, Groq uses an OpenAI-compatible API. So the code here
follows the exact same pattern as openrouter.py — we reuse the `openai`
SDK with a different base_url.

NOTICE THE PATTERN: This file is structurally identical to openrouter.py.
That's intentional. When multiple APIs follow the same standard, your
code for each one should look nearly the same. The differences are:
    1. base_url → "https://api.groq.com/openai/v1"
    2. api_key → Your Groq key
    3. Available models → Different set of free models

WHY "groq_provider.py" AND NOT "groq.py"?
    There's an official `groq` Python package. If we named our file
    "groq.py", Python might confuse our file with the package. Adding
    "_provider" avoids this naming collision entirely.
"""

from openai import OpenAI

from config import settings
from llm_providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
)


# ── Free Models on Groq ───────────────────────────────────────────────
# Groq's free tier models. These are very fast due to Groq's custom chips.

GROQ_FREE_MODELS = [
    (
        "llama-3.1-8b-instant",
        "Llama 3.1 8B (Groq)",
        "Blazing fast on Groq's LPU. Great for quick tasks.",
    ),
    (
        "gemma2-9b-it",
        "Gemma 2 9B (Groq)",
        "Google's Gemma model, accelerated by Groq hardware.",
    ),
    (
        "mixtral-8x7b-32768",
        "Mixtral 8x7B (Groq)",
        "Mixture-of-experts model. Strong reasoning, 32K context.",
    ),
]


class GroqProvider(BaseLLMProvider):
    """
    LLM provider that connects to Groq's API.

    Groq is OpenAI-compatible, so the implementation is nearly identical
    to OpenRouterProvider. The key differences are the base_url and the
    available models.
    """

    def __init__(self) -> None:
        """
        Initialize the Groq provider.

        Same pattern as OpenRouter: create an OpenAI client with
        Groq's endpoint URL and API key.
        """
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key or "",
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
        Send a conversation to a Groq model and get a response.

        Identical flow to OpenRouter — convert messages, call API,
        wrap response. The OpenAI SDK handles all the HTTP details.
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
            provider="groq",
            usage=usage,
            raw_response=response,
        )

    def get_available_models(self) -> list[ModelInfo]:
        """Return the list of free models available on Groq."""
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="groq",
                description=desc,
            )
            for model_id, name, desc in GROQ_FREE_MODELS
        ]

    def is_available(self) -> bool:
        """Check if Groq is configured with an API key."""
        return bool(settings.groq_api_key)
