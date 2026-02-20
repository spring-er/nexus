"""
llm_providers/gemini_provider.py — Google Gemini LLM Provider

Google AI Studio (aistudio.google.com) provides free access to Gemini
models. Unlike OpenRouter and Groq, Gemini does NOT use the OpenAI API
format. It has its own SDK with a different message structure.

THIS IS WHY THE BASE CLASS MATTERS:
    OpenRouter, Groq, and Ollama all used the `openai` SDK (same pattern).
    Gemini uses a completely different SDK. But because all providers
    inherit from BaseLLMProvider, the rest of our app doesn't care —
    it just calls `provider.chat(messages)` the same way every time.

KEY DIFFERENCES FROM OPENAI-COMPATIBLE PROVIDERS:
    1. Uses `google-genai` SDK (not `openai`)
    2. Roles: "user" and "model" (not "user" and "assistant")
    3. System prompt: Passed via `system_instruction` in the config,
       NOT as the first message in the conversation
    4. Message format: Uses Content/Parts objects

FREE TIER:
    Gemini Flash models are free with generous rate limits
    (15 requests/minute, 1M tokens/minute as of 2024).
"""

from config import settings
from llm_providers.base import (
    BaseLLMProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
)

# Lazy import: The Google SDK has heavy dependencies.
# If those fail to load, we just disable the Gemini provider.
try:
    from google import genai
    from google.genai import types

    GEMINI_SDK_AVAILABLE = True
except BaseException:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GEMINI_SDK_AVAILABLE = False


# ── Free Gemini Models ─────────────────────────────────────────────────
# Gemini Flash models are free tier. "Flash" = optimized for speed.

GEMINI_FREE_MODELS = [
    (
        "gemini-2.0-flash",
        "Gemini 2.0 Flash",
        "Google's latest fast model. Great balance of speed and quality.",
    ),
    (
        "gemini-1.5-flash",
        "Gemini 1.5 Flash",
        "Very fast, 1M token context window. Good for long documents.",
    ),
]


class GeminiProvider(BaseLLMProvider):
    """
    LLM provider that connects to Google's Gemini API.

    Uses the new `google-genai` SDK (replacing the deprecated
    `google-generativeai` package). The main job of this class is to
    translate between our standard ChatMessage format and Gemini's format.

    Attributes:
        _client: The google.genai.Client instance, or None if not configured.
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini provider.

        The new SDK uses an explicit Client object initialized with an
        API key, replacing the old global genai.configure() pattern.
        """
        self._client = None
        if GEMINI_SDK_AVAILABLE and settings.google_api_key:
            self._client = genai.Client(api_key=settings.google_api_key)

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a conversation to a Gemini model and get a response.

        This method handles the translation between formats:

        OUR FORMAT (OpenAI-style):
            [
                {"role": "system", "content": "You are helpful..."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]

        GEMINI FORMAT:
            config.system_instruction = "You are helpful..."
            history = [
                Content(role="user", parts=[Part(text="Hello")]),
                Content(role="model", parts=[Part(text="Hi there!")]),
            ]
            + send "How are you?" as the new message
        """
        # Step 1: Extract the system prompt (if any).
        system_instruction = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                conversation_messages.append(msg)

        # Step 2: Build the generation config.
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Step 3: Convert conversation history to Gemini's Content format.
        # Gemini uses "model" where OpenAI uses "assistant".
        gemini_history = []
        for msg in conversation_messages[:-1]:  # All messages except the last
            gemini_role = "model" if msg.role == "assistant" else "user"
            gemini_history.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(msg.content)],
                )
            )

        # Step 4: Create a chat session and send the last message.
        chat_session = self._client.chats.create(
            model=model,
            config=config,
            history=gemini_history,
        )

        last_message = conversation_messages[-1].content if conversation_messages else ""
        response = chat_session.send_message(last_message)

        # Step 5: Extract usage stats.
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return ChatResponse(
            content=response.text,
            model=model,
            provider="gemini",
            usage=usage,
            raw_response=response,
        )

    def get_available_models(self) -> list[ModelInfo]:
        """Return the list of free Gemini models."""
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="gemini",
                description=desc,
            )
            for model_id, name, desc in GEMINI_FREE_MODELS
        ]

    def is_available(self) -> bool:
        """Check if the Gemini SDK loaded and a Google API key is set."""
        return GEMINI_SDK_AVAILABLE and self._client is not None
