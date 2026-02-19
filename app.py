"""
app.py — Main Entry Point for Nexus AI

Run this file to start the application:
    python app.py

This module:
    1. Initializes the ProviderManager (connects to all LLM providers)
    2. Creates the Gradio web interface with tabs
    3. Launches the server at http://localhost:7860

ARCHITECTURE:
    app.py is the "glue" that wires everything together. It doesn't
    contain business logic — it just creates the components and
    assembles them into a running application. This separation makes
    each piece independently testable.
"""

import gradio as gr

from config import settings
from llm_providers.router import ProviderManager
from ui.chat_tab import create_chat_tab


def create_app() -> gr.Blocks:
    """
    Build the complete Gradio application.

    We separate app creation from app launching so that:
    - Tests can create the app without launching a server
    - We can reuse this function in different contexts

    Returns:
        A gr.Blocks application ready to launch.
    """
    # Step 1: Initialize the provider manager.
    # This creates all LLM provider instances and builds the model registry.
    print(f"Initializing {settings.app_name}...")
    manager = ProviderManager()

    # Log provider status to the terminal
    for name, available in manager.get_provider_status().items():
        status = "READY" if available else "not configured"
        print(f"  {name}: {status}")

    available_count = len(manager.get_available_models())
    total_count = len(manager.get_all_models())
    print(f"  Models: {available_count} available / {total_count} total")

    # Step 2: Build the Gradio interface.
    # gr.Blocks is the top-level container. We'll add tabs for each
    # major feature (chat, knowledge base, agents, etc.).
    with gr.Blocks(
        title="Nexus AI",
        theme=gr.themes.Soft(),
    ) as app:
        # App header
        gr.Markdown(
            "# Nexus AI\n"
            "*Your personal AI research & productivity assistant*"
        )

        # Tabbed interface — each tab is a major feature.
        # We start with just Chat. More tabs will be added in later phases:
        #   Phase 2: Knowledge Base tab
        #   Phase 3: Agents tab
        #   Phase 4: Tools tab
        #   Phase 6: Dashboard tab
        with gr.Tabs():
            with gr.Tab("Chat"):
                create_chat_tab(manager)

            # Placeholder tabs for future phases (shows the roadmap)
            with gr.Tab("Knowledge Base"):
                gr.Markdown(
                    "### Coming in Phase 2\n"
                    "Upload documents, PDFs, and URLs to build your "
                    "personal knowledge base. Ask questions that are "
                    "answered from your own data using RAG."
                )

            with gr.Tab("Agents"):
                gr.Markdown(
                    "### Coming in Phase 3\n"
                    "AI agents that can research, write, and take "
                    "actions on your behalf using LangGraph."
                )

            with gr.Tab("Tools"):
                gr.Markdown(
                    "### Coming in Phase 4\n"
                    "Connect external tools and services via MCP "
                    "(Model Context Protocol) and Composio."
                )

            with gr.Tab("Dashboard"):
                gr.Markdown(
                    "### Coming in Phase 6\n"
                    "Monitor token usage, costs, and performance "
                    "across all models and providers."
                )

    return app


def main() -> None:
    """Create and launch the Nexus AI application."""
    app = create_app()

    # Launch the Gradio server.
    # - server_name="0.0.0.0" makes it accessible from other devices on your network
    # - share=False means no public URL (set True to share via Gradio's tunnel)
    print(f"\nStarting {settings.app_name} server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
