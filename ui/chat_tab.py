"""
ui/chat_tab.py — Chat Interface Tab for Gradio UI

This module builds the main chat tab where users:
    1. Select a model from a dropdown
    2. Adjust temperature and max tokens
    3. Chat with the selected LLM
    4. See conversation history

GRADIO CONCEPTS USED:
    - gr.Blocks: A layout container. Unlike gr.Interface (which is simpler),
      Blocks gives us full control over the UI layout.
    - gr.ChatInterface: A pre-built chat component that handles message
      display, input, history, and streaming.
    - gr.State: Stores data that persists across interactions for a single
      user session. We use it to keep conversation history.
    - gr.Dropdown: A dropdown selector for choosing models.

HOW THE CHAT FLOW WORKS:
    1. User types a message and hits Enter
    2. Gradio calls our `respond()` function with the message + history
    3. We prepend the system prompt, add history, add the new message
    4. We send everything to the selected model via ProviderManager
    5. The response is displayed in the chat window
"""

import gradio as gr

from llm_providers.base import ChatMessage
from llm_providers.router import ProviderManager
from llm_providers.system_prompts import DEFAULT_SYSTEM_PROMPT


def create_chat_tab(manager: ProviderManager) -> gr.Blocks:
    """
    Build and return the Chat tab as a Gradio Blocks component.

    Args:
        manager: The ProviderManager instance (our unified LLM interface).
                 We pass it in rather than creating it here so the whole
                 app shares one instance.

    Returns:
        A gr.Blocks component containing the full chat interface.
    """
    # Get available models for the dropdown.
    # Format: "Display Name (provider)" as the label, model_id as the value.
    available_models = manager.get_available_models()
    all_models = manager.get_all_models()

    # Build dropdown choices: list of (label, value) tuples.
    # If no models are available (no API keys), show all models as a hint.
    models_to_show = available_models if available_models else all_models
    model_choices = [
        (f"{m.name} [{m.provider}]", m.id) for m in models_to_show
    ]

    # Default model selection
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
    ) -> str:
        """
        Process a user message and return the AI's response.

        This function is called by Gradio every time the user sends a message.

        Args:
            message: The user's new message.
            history: Previous messages as a list of {"role": ..., "content": ...} dicts.
                     Gradio manages this automatically.
            model_id: The selected model's ID from the dropdown.
            temperature: Creativity slider value.
            max_tokens: Maximum response length slider value.
            system_prompt: The system prompt text (editable by user).

        Returns:
            The AI's response as a string.
        """
        # Step 1: Build the full message list for the API.
        # Start with the system prompt, then add conversation history,
        # then add the new user message.
        messages = [ChatMessage(role="system", content=system_prompt)]

        # Add conversation history.
        # Gradio's type="messages" format gives us dicts with "role" and "content".
        for msg in history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        # Add the new user message
        messages.append(ChatMessage(role="user", content=message))

        # Step 2: Send to the LLM via the ProviderManager.
        try:
            response = manager.chat(
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content

        except Exception as e:
            # If something goes wrong (bad key, network error, etc.),
            # show the error to the user instead of crashing.
            return f"**Error:** {str(e)}\n\nPlease check your API key and try again."

    # ── Build the Gradio Layout ────────────────────────────────────────
    # We use gr.Blocks for full layout control.

    with gr.Blocks() as chat_tab:
        gr.Markdown("## Chat with AI")
        gr.Markdown(
            "Select a model and start chatting. "
            "Adjust temperature (creativity) and max tokens (response length) below."
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
                    # type="messages" gives us the modern dict format
                    # with {"role": "user"/"assistant", "content": "..."}
                    type="messages",
                    additional_inputs=[
                        model_dropdown,
                        temperature_slider,
                        max_tokens_slider,
                        system_prompt_box,
                    ],
                    title=None,  # We already have a title above
                    retry_btn="Retry",
                    undo_btn="Undo",
                    clear_btn="Clear Chat",
                )

    return chat_tab
