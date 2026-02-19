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
    1. Uses `google.generativeai` SDK (not `openai`)
    2. Roles: "user" and "model" (not "user" and "assistant")
    3. System prompt: Passed via `system_instruction` parameter, NOT
       as the first message in the conversation
    4. Message format: Uses "parts" instead of "content"

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

# Lazy import: The Google SDK has heavy dependencies (grpc, cryptography).
# If those fail to load (e.g., missing system libraries), we don't want
# the entire app to crash — we just disable the Gemini provider.
try:
    import google.generativeai as genai

    GEMINI_SDK_AVAILABLE = True
except BaseException:
    # We catch BaseException (not just Exception) because some SDK failures
    # are Rust-level panics that don't inherit from Exception.
    # On your local machine with a proper Python environment, this should
    # never trigger — it's a safety net for unusual environments.
    genai = None  # type: ignore[assignment]
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

    This provider uses Google's own SDK, which has a different API
    structure than OpenAI-compatible providers. The main job of this
    class is to translate between our standard ChatMessage format
    and Gemini's format.

    Attributes:
        _configured: Whether the API key has been set up.
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini provider.

        google.generativeai uses a global configuration (genai.configure)
        rather than a client instance. This is a design choice by Google —
        you configure the API key once, then create model instances.
        """
        self._configured = False
        if GEMINI_SDK_AVAILABLE and settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self._configured = True

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
            system_instruction = "You are helpful..."
            history = [
                {"role": "user", "parts": ["Hello"]},
                {"role": "model", "parts": ["Hi there!"]},
            ]
            + send "How are you?" as the new message
        """
        # Step 1: Extract the system prompt (if any).
        # In OpenAI format, the system prompt is the first message with role="system".
        # Gemini handles it differently — as a separate parameter.
        system_instruction = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                conversation_messages.append(msg)

        # Step 2: Create the Gemini model instance.
        # We pass the system prompt here, NOT in the message list.
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

        # Step 3: Convert conversation history to Gemini's format.
        # Gemini uses "model" where OpenAI uses "assistant".
        # Gemini uses "parts" (a list) where OpenAI uses "content" (a string).
        gemini_history = []
        for msg in conversation_messages[:-1]:  # All messages except the last
            gemini_role = "model" if msg.role == "assistant" else "user"
            gemini_history.append({
                "role": gemini_role,
                "parts": [msg.content],
            })

        # Step 4: Start a chat session with the history and send the last message.
        # Gemini's chat.send_message() appends to the history automatically.
        chat_session = gemini_model.start_chat(history=gemini_history)

        # The last message is what we're actually asking
        last_message = conversation_messages[-1].content if conversation_messages else ""
        response = chat_session.send_message(last_message)

        # Step 5: Extract usage stats.
        # Gemini provides these through response.usage_metadata.
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
        return GEMINI_SDK_AVAILABLE and self._configured
