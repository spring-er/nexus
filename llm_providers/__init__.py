"""
llm_providers — LLM Provider Wrappers for Nexus AI

This package contains one module per LLM provider (OpenRouter, Groq,
Ollama, Gemini). Each module wraps the provider's API behind a common
interface defined in base.py, so the rest of the app doesn't need to
know which provider it's talking to.
"""
