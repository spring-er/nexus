"""
llm_providers/openrouter.py — OpenRouter LLM Provider

OpenRouter (openrouter.ai) is a unified API gateway that provides access
to many LLM models through a single API key. It uses the OpenAI-compatible
API format, so we can use the standard `openai` Python SDK — we just
change the base_url to point at OpenRouter instead of OpenAI.

FREE MODELS: OpenRouter offers several models at no cost. Their model IDs
end with ":free" (e.g., "meta-llama/llama-3.1-8b-instruct:free"). These
have rate limits but are perfect for learning and development.

HOW IT WORKS:
    1. We create an OpenAI client but point it at OpenRouter's URL
    2. We send messages in the standard OpenAI chat format
    3. OpenRouter routes our request to the chosen model
    4. We get back a response in the standard OpenAI format

WHY THIS APPROACH:
    - We don't need a separate "openrouter" library
    - The same code pattern works for Groq (also OpenAI-compatible)
    - If OpenAI changes their SDK, both providers benefit from the update
"""

from openai import OpenAI

from config import settings
from llm_providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
)


# ── Free Models on OpenRouter ──────────────────────────────────────────
# These are the models we'll offer in the dropdown. All are free tier.
# Each tuple: (model_id, display_name, short_description)

OPENROUTER_FREE_MODELS = [
    (
        "meta-llama/llama-3.1-8b-instruct:free",
        "Llama 3.1 8B",
        "Meta's fast, general-purpose model. Great for everyday tasks.",
    ),
    (
        "google/gemma-2-9b-it:free",
        "Gemma 2 9B",
        "Google's efficient open model. Good at following instructions.",
    ),
    (
        "mistralai/mistral-7b-instruct:free",
        "Mistral 7B",
        "Fast European model. Strong reasoning for its size.",
    ),
    (
        "qwen/qwen-2.5-7b-instruct:free",
        "Qwen 2.5 7B",
        "Alibaba's multilingual model. Excellent at code and math.",
    ),
]


class OpenRouterProvider(BaseLLMProvider):
    """
    LLM provider that connects to OpenRouter's API.

    OpenRouter is OpenAI-compatible, so we reuse the `openai` SDK
    with a different base_url. This is the same pattern used by
    dozens of AI providers in the industry.

    Attributes:
        client: An OpenAI SDK client configured for OpenRouter's endpoint.
    """

    def __init__(self) -> None:
        """
        Initialize the OpenRouter provider.

        We create the OpenAI client here but point it at OpenRouter's URL.
        The three key differences from a real OpenAI client:
            1. base_url → "https://openrouter.ai/api/v1"
            2. api_key → Your OpenRouter key (not an OpenAI key)
            3. default_headers → Identifies our app to OpenRouter
        """
        self.client = OpenAI(
            # This is the magic: same SDK, different endpoint
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key or "",
            default_headers={
                # OpenRouter recommends identifying your app
                "HTTP-Referer": "https://github.com/nexus-ai",
                "X-Title": "Nexus AI",
            },
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
        Send a conversation to an OpenRouter model and get a response.

        This method:
            1. Converts our ChatMessage objects to the dict format the API expects
            2. Calls the OpenRouter API (via the OpenAI SDK)
            3. Wraps the response in our standard ChatResponse format

        Args:
            messages: Conversation history (system + user + assistant messages).
            model: OpenRouter model ID (e.g., "meta-llama/llama-3.1-8b-instruct:free").
            temperature: Randomness control (0.0-1.0).
            max_tokens: Maximum response length in tokens.

        Returns:
            ChatResponse with the model's reply and usage statistics.

        Raises:
            Exception: If the API call fails (bad key, rate limit, network error).
        """
        # Convert our ChatMessage objects to dicts the OpenAI SDK expects.
        # The API wants: [{"role": "user", "content": "Hello"}]
        api_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Make the API call. This sends an HTTP POST to OpenRouter.
        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract the generated text from the response.
        # The OpenAI format nests it under: response.choices[0].message.content
        content = response.choices[0].message.content or ""

        # Extract token usage stats (how many tokens were used).
        # This is important for cost tracking later in Phase 6.
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
            provider="openrouter",
            usage=usage,
            raw_response=response,
        )

    def get_available_models(self) -> list[ModelInfo]:
        """
        Return the list of free models available on OpenRouter.

        We use a hardcoded list rather than querying the API because:
        1. The API returns 100+ models, most of which are paid
        2. We want to curate only the free, reliable ones
        3. It's faster (no network call needed)
        """
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="openrouter",
                description=desc,
            )
            for model_id, name, desc in OPENROUTER_FREE_MODELS
        ]

    def is_available(self) -> bool:
        """
        Check if OpenRouter is configured with an API key.

        Returns True if the OPENROUTER_API_KEY environment variable
        is set (non-empty). The key's validity is only checked when
        we actually make an API call.
        """
        return bool(settings.openrouter_api_key)
