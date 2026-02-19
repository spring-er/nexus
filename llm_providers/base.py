"""
llm_providers/base.py — Abstract Base Class for All LLM Providers

This module defines the "contract" that every LLM provider must follow.
By using an abstract base class (ABC), we guarantee that OpenRouter, Groq,
Ollama, and Gemini all implement the same methods. This means the rest of
the app can work with ANY provider without knowing the details.

DESIGN PATTERN: Strategy Pattern
    - Define a common interface (BaseLLMProvider)
    - Each provider implements it differently
    - The app swaps providers at runtime without code changes

KEY CONCEPTS:
    - ABC (Abstract Base Class): A class you can't instantiate directly.
      It exists only to be inherited. If a subclass forgets to implement
      a required method, Python raises an error at class creation time.
    - @abstractmethod: Marks a method that subclasses MUST implement.
    - dataclass: A Python decorator that auto-generates __init__, __repr__,
      etc. Less boilerplate than writing them by hand.
    - Type hints: Every parameter and return value is annotated with its
      type. This helps your IDE catch bugs and makes code self-documenting.

USAGE:
    class MyProvider(BaseLLMProvider):
        def chat(self, messages, **kwargs):
            # ... call the API ...
            return ChatResponse(content="Hello!", model="my-model", ...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ── Data Classes ────────────────────────────────────────────────────────
# These are simple containers for structured data. Using @dataclass
# means Python auto-generates __init__ and __repr__ for us.


@dataclass
class ChatMessage:
    """
    A single message in a conversation.

    Attributes:
        role: Who sent this message. One of:
              - "system": Instructions for the AI's behavior
              - "user": The human's message
              - "assistant": The AI's response
        content: The text of the message.
    """

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ChatResponse:
    """
    The result returned after calling an LLM.

    We wrap the raw API response in this standard format so the rest
    of the app doesn't need to parse different response formats from
    different providers.

    Attributes:
        content: The text the LLM generated.
        model: Which model actually produced the response.
        provider: Which provider served the request (e.g., "openrouter").
        usage: Token usage stats (input tokens, output tokens, total).
               Not all providers return this, so it's optional.
        raw_response: The original API response object, in case we need
                      to access provider-specific fields later.
    """

    content: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    raw_response: Optional[object] = None


@dataclass
class ModelInfo:
    """
    Metadata about a single model available through a provider.

    This powers the model selector dropdown in the UI — we need to
    know each model's ID (for the API call) and its display name
    (for the user to read).

    Attributes:
        id: The model identifier used in API calls (e.g., "meta-llama/llama-3.1-8b-instruct:free").
        name: A human-friendly display name (e.g., "Llama 3.1 8B").
        provider: Which provider serves this model (e.g., "openrouter").
        description: Optional short description of the model's strengths.
    """

    id: str
    name: str
    provider: str
    description: str = ""


# ── Abstract Base Class ─────────────────────────────────────────────────


class BaseLLMProvider(ABC):
    """
    Abstract base class that all LLM providers must inherit from.

    Any class that inherits from BaseLLMProvider MUST implement:
        - chat(): Send messages and get a response
        - get_available_models(): List which models this provider offers
        - is_available(): Check if the provider is configured and ready

    If a subclass forgets to implement any of these, Python will raise
    a TypeError when you try to create an instance of it.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a conversation to the LLM and get a response.

        Args:
            messages: The conversation history as a list of ChatMessage objects.
                      Typically starts with a system message, then alternates
                      between user and assistant messages.
            model: The model ID to use (e.g., "meta-llama/llama-3.1-8b-instruct:free").
            temperature: Controls randomness. 0.0 = deterministic, 1.0 = creative.
                         Default 0.7 is a good balance.
            max_tokens: Maximum number of tokens the model can generate.
                        1024 tokens is roughly 750 words.
            **kwargs: Extra provider-specific options (passed through to the API).

        Returns:
            A ChatResponse with the generated text, model info, and usage stats.

        Raises:
            Exception: If the API call fails (network error, invalid key, etc.).
        """
        ...

    @abstractmethod
    def get_available_models(self) -> list[ModelInfo]:
        """
        Return the list of models this provider offers.

        This is used to populate the model selector dropdown in the UI.
        Each provider defines its own list of free models.

        Returns:
            A list of ModelInfo objects describing available models.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this provider is configured and ready to use.

        A provider is "available" if it has a valid API key configured
        (or, for Ollama, if the local server is reachable).

        Returns:
            True if the provider can accept requests, False otherwise.
        """
        ...
