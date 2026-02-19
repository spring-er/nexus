"""
llm_providers/router.py — Provider Manager (Unified LLM Interface)

This module is the SINGLE POINT OF CONTACT between the rest of the app
and all LLM providers. Instead of the UI knowing about OpenRouter, Groq,
Ollama, and Gemini individually, it just calls:

    manager = ProviderManager()
    response = manager.chat(messages, "meta-llama/llama-3.1-8b-instruct:free")

The manager figures out which provider to use based on the model ID.

DESIGN PATTERN: Facade Pattern
    - Hides the complexity of 4 different providers behind one simple interface
    - The UI doesn't need to know which provider handles which model
    - Adding a new provider only requires changes HERE, not in the UI

HOW ROUTING WORKS:
    When we initialize, we register every model from every provider in a
    lookup dictionary: { model_id → provider_instance }. When chat() is
    called with a model_id, we look it up and route to the correct provider.
"""

from llm_providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
)
from llm_providers.openrouter import OpenRouterProvider
from llm_providers.groq_provider import GroqProvider
from llm_providers.ollama_provider import OllamaProvider
from llm_providers.gemini_provider import GeminiProvider


class ProviderManager:
    """
    Unified interface for all LLM providers.

    This class:
        1. Initializes all providers at startup
        2. Collects available models from each provider
        3. Routes chat requests to the correct provider based on model ID

    Attributes:
        providers: Dict mapping provider name → provider instance.
        model_registry: Dict mapping model_id → provider instance.
                        This is the lookup table for routing.
        all_models: List of all ModelInfo objects across all providers.
    """

    def __init__(self) -> None:
        """
        Initialize all providers and build the model routing table.

        We create an instance of each provider, check which ones are
        available (have API keys configured), and register their models.
        Unavailable providers are kept but their models are marked
        accordingly.
        """
        # Step 1: Create all provider instances.
        # Even if a provider isn't available (no API key), we still create it
        # so we can list its models in the UI (shown as disabled).
        self.providers: dict[str, BaseLLMProvider] = {
            "openrouter": OpenRouterProvider(),
            "groq": GroqProvider(),
            "ollama": OllamaProvider(),
            "gemini": GeminiProvider(),
        }

        # Step 2: Build the routing table.
        # For each available provider, map its model IDs to the provider instance.
        # This lets us do: model_registry["llama-3.1-8b-instant"] → GroqProvider
        self.model_registry: dict[str, BaseLLMProvider] = {}
        self.all_models: list[ModelInfo] = []

        for provider_name, provider in self.providers.items():
            models = provider.get_available_models()
            self.all_models.extend(models)

            # Only register models from available providers for actual routing
            if provider.is_available():
                for model in models:
                    self.model_registry[model.id] = provider

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat request to the appropriate provider.

        This is the main method the rest of the app calls. It:
            1. Looks up which provider handles the given model
            2. Forwards the request to that provider
            3. Returns the standardized ChatResponse

        Args:
            messages: Conversation history.
            model: Model ID (must be registered in model_registry).
            temperature: Randomness control (0.0-1.0).
            max_tokens: Maximum response length.

        Returns:
            ChatResponse from the appropriate provider.

        Raises:
            ValueError: If the model ID isn't recognized or its provider
                        isn't available.
        """
        provider = self.model_registry.get(model)

        if provider is None:
            # Give a helpful error message listing what IS available
            available = list(self.model_registry.keys())
            raise ValueError(
                f"Model '{model}' is not available. "
                f"Available models: {available}"
            )

        return provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_available_models(self) -> list[ModelInfo]:
        """
        Return models from available (configured) providers only.

        This is used to populate the model selector dropdown. We only
        show models that the user can actually use right now.
        """
        return [
            model
            for model in self.all_models
            if model.id in self.model_registry
        ]

    def get_all_models(self) -> list[ModelInfo]:
        """
        Return ALL models from ALL providers (including unavailable ones).

        Useful for showing the user what they COULD use if they added
        more API keys. The UI can show these as disabled/grayed out.
        """
        return self.all_models

    def get_provider_status(self) -> dict[str, bool]:
        """
        Return the availability status of each provider.

        Used in the UI to show which providers are active (green)
        vs unconfigured (gray).

        Returns:
            Dict like {"openrouter": True, "groq": False, ...}
        """
        return {
            name: provider.is_available()
            for name, provider in self.providers.items()
        }

    def get_default_model(self) -> str | None:
        """
        Return the first available model ID, or None if nothing is available.

        Used to set the initial selection in the model dropdown.
        """
        available = self.get_available_models()
        return available[0].id if available else None
