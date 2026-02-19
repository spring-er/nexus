"""
ui/chat_tab.py — Chat Interface Tab for Gradio UI

This module builds the main chat tab where users:
    1. Select a model from a dropdown
    2. Adjust temperature and max tokens
    3. Chat with the selected LLM
    4. Optionally use RAG to answer from their knowledge base
    5. See conversation history with source citations

GRADIO CONCEPTS USED:
    - gr.Blocks: A layout container for full control over the UI layout.
    - gr.ChatInterface: Pre-built chat component with message display,
      input, history, and retry/undo/clear buttons.
    - gr.Dropdown: Model selector.
    - gr.Checkbox: Toggle RAG on/off.

HOW RAG-ENHANCED CHAT WORKS:
    1. User types a question
    2. If RAG is enabled AND the knowledge base has documents:
       a. Search the vector store for relevant chunks
       b. Build a context-augmented prompt (question + relevant docs)
       c. Use the RAG system prompt instead of the default one
    3. Send to the LLM
    4. Append source citations to the response
"""

import gradio as gr

from llm_providers.base import ChatMessage
from llm_providers.router import ProviderManager
from llm_providers.system_prompts import DEFAULT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from rag.query_engine import build_rag_context, format_rag_prompt, format_citations
from monitoring.tracker import track_llm_call, Timer


def create_chat_tab(manager: ProviderManager) -> gr.Blocks:
    """
    Build and return the Chat tab as a Gradio Blocks component.

    Args:
        manager: The ProviderManager instance (our unified LLM interface).

    Returns:
        A gr.Blocks component containing the full chat interface.
    """
    available_models = manager.get_available_models()
    all_models = manager.get_all_models()

    models_to_show = available_models if available_models else all_models
    model_choices = [
        (f"{m.name} [{m.provider}]", m.id) for m in models_to_show
    ]

    default_model = manager.get_default_model()
    if not default_model and model_choices:
        default_model = model_choices[0][1]

    def respond(
        message: str,
        history: list[dict],
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        use_rag: bool,
    ) -> str:
        """
        Process a user message and return the AI's response.

        Now RAG-aware: if RAG is enabled, the function searches the
        knowledge base for relevant context and includes it in the prompt.

        Args:
            message: The user's new message.
            history: Previous messages as dicts with "role" and "content".
            model_id: Selected model ID.
            temperature: Creativity slider value.
            max_tokens: Max response length.
            system_prompt: System prompt text.
            use_rag: Whether to search the knowledge base.

        Returns:
            The AI's response, potentially with source citations appended.
        """
        # Step 1: Optionally enhance the message with RAG context.
        # If RAG is enabled, search the knowledge base and inject
        # relevant document chunks into the prompt.
        citations_text = ""
        active_system_prompt = system_prompt

        if use_rag:
            context, sources = build_rag_context(message, top_k=5)
            if context:
                # We have relevant documents — use the RAG system prompt
                # and augment the user's message with context
                message = format_rag_prompt(message, context)
                active_system_prompt = RAG_SYSTEM_PROMPT
                citations_text = format_citations(sources)

        # Step 2: Build the full message list for the API.
        messages = [ChatMessage(role="system", content=active_system_prompt)]

        for msg in history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        messages.append(ChatMessage(role="user", content=message))

        # Step 3: Send to the LLM via the ProviderManager.
        # Wrapped with monitoring to track tokens, cost, and latency.
        try:
            with Timer() as t:
                response = manager.chat(
                    messages=messages,
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Log the interaction to our monitoring system
            feature = "rag_chat" if (use_rag and citations_text) else "chat"
            track_llm_call(
                model=model_id,
                provider=response.provider,
                input_tokens=response.usage.get("prompt_tokens", 0),
                output_tokens=response.usage.get("completion_tokens", 0),
                latency_ms=t.elapsed_ms,
                success=True,
                feature=feature,
            )

            # Append citations if RAG was used
            return response.content + citations_text

        except Exception as e:
            # Log failures too — they help with debugging
            track_llm_call(
                model=model_id,
                provider="unknown",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                success=False,
                error=str(e),
                feature="chat",
            )
            return f"**Error:** {str(e)}\n\nPlease check your API key and try again."

    # ── Build the Gradio Layout ────────────────────────────────────────

    with gr.Blocks() as chat_tab:
        gr.Markdown("## Chat with AI")
        gr.Markdown(
            "Select a model and start chatting. "
            "Enable **Use Knowledge Base** to answer from your uploaded documents."
        )

        with gr.Row():
            # Left column: Settings panel
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Model",
                    info="Choose which AI model to chat with",
                    interactive=True,
                )

                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="0 = focused & deterministic, 1.5 = creative & random",
                )

                max_tokens_slider = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=1024,
                    step=64,
                    label="Max Tokens",
                    info="Maximum response length (~750 words per 1024 tokens)",
                )

                # RAG toggle — NEW in Phase 2
                use_rag_checkbox = gr.Checkbox(
                    value=True,
                    label="Use Knowledge Base",
                    info="Search your uploaded documents to enhance answers",
                )

                # Collapsible system prompt editor
                with gr.Accordion("System Prompt", open=False):
                    system_prompt_box = gr.Textbox(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="System Prompt",
                        lines=10,
                        info="Instructions that define the AI's personality and behavior",
                    )

                # Provider status display
                with gr.Accordion("Provider Status", open=False):
                    status = manager.get_provider_status()
                    status_text = "\n".join(
                        f"{'✓' if available else '✗'} {name}"
                        for name, available in status.items()
                    )
                    gr.Markdown(f"```\n{status_text}\n```")

            # Right column: Chat interface (takes more space)
            with gr.Column(scale=3):
                chat_interface = gr.ChatInterface(
                    fn=respond,
                    type="messages",
                    additional_inputs=[
                        model_dropdown,
                        temperature_slider,
                        max_tokens_slider,
                        system_prompt_box,
                        use_rag_checkbox,
                    ],
                    title=None,
                    retry_btn="Retry",
                    undo_btn="Undo",
                    clear_btn="Clear Chat",
                )

    return chat_tab
