"""
app.py — Main Entry Point for Nexus AI

Run this file to start the application:
    python app.py

This module creates and launches the Gradio web interface.
We'll build out the full UI in Phase 1, Step 9.
"""

from config import settings


def main() -> None:
    """Start the Nexus AI application."""
    print(f"Starting {settings.app_name}...")
    print(f"Debug mode: {settings.debug}")

    # Check which providers have API keys configured
    providers_status = {
        "OpenRouter": bool(settings.openrouter_api_key),
        "Groq": bool(settings.groq_api_key),
        "Google Gemini": bool(settings.google_api_key),
        "Ollama": True,  # Local, always available if running
    }

    print("\nProvider status:")
    for provider, available in providers_status.items():
        status = "READY" if available else "No API key"
        print(f"  {provider}: {status}")

    print("\nProject setup complete! Ready to build providers.")


if __name__ == "__main__":
    main()
