"""
agents/tools.py — Agent Tools for Nexus AI

Tools are functions that an AI agent can call to take actions in the world.
The agent doesn't execute tools directly — it outputs a structured "tool call"
(function name + arguments), and our code executes it and returns the result.

HOW TOOL CALLING WORKS:
    1. We define tools with a name, description, and parameter schema
    2. The LLM sees these definitions and decides when to use them
    3. When the LLM outputs a tool call, LangGraph catches it
    4. LangGraph runs the tool function with the provided arguments
    5. The result is sent back to the LLM as an "observation"
    6. The LLM reads the observation and decides its next step

TOOLS IN THIS MODULE:
    - web_search: Search the web using DuckDuckGo (free, no API key)
    - scrape_webpage: Fetch and extract text from a URL
    - knowledge_base_search: Search the user's uploaded documents
    - summarize_text: Condense long text into key points

WHY DUCKDUCKGO?
    Google Search API costs money and requires setup. DuckDuckGo's API
    is free, requires no API key, and returns good results. For a
    learning project, it's the perfect choice.

LANGCHAIN TOOLS:
    We use the @tool decorator from langchain-core. This automatically:
    - Generates the JSON schema from the function signature & docstring
    - Handles serialization/deserialization of arguments
    - Makes the tool compatible with LangGraph's agent framework
"""

from langchain_core.tools import tool

from rag.vector_store import search as vector_search


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Use this when you need up-to-date information, facts, news, or
    information that may not be in the knowledge base.

    Args:
        query: The search query string.

    Returns:
        A formatted string of search results with titles, snippets, and URLs.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            # Get top 5 results. Each result has: title, body, href
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        # Format results for the LLM to read
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r['title']}\n"
                f"    {r['body']}\n"
                f"    URL: {r['href']}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def scrape_webpage(url: str) -> str:
    """Fetch a webpage and extract its text content.

    Use this to read the full content of a specific URL. Useful after
    web_search returns interesting URLs that need deeper reading.

    Args:
        url: The full URL to fetch (must start with http:// or https://).

    Returns:
        The extracted text content of the page (truncated to 3000 chars).
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        response = httpx.get(url, follow_redirects=True, timeout=15.0)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid overwhelming the LLM context
        if len(text) > 3000:
            text = text[:3000] + "\n\n[... truncated ...]"

        return text if text.strip() else "No readable text found on this page."

    except Exception as e:
        return f"Failed to fetch {url}: {str(e)}"


@tool
def knowledge_base_search(query: str) -> str:
    """Search the user's personal knowledge base for relevant information.

    Use this to find information from documents the user has uploaded
    (PDFs, text files, web pages, YouTube transcripts).

    Args:
        query: The search query describing what information you need.

    Returns:
        Relevant text chunks from the knowledge base with source citations.
    """
    try:
        results = vector_search(query, top_k=3)

        if not results:
            return "No relevant documents found in the knowledge base."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] Source: {r['title']} ({r['source']})\n"
                f"    {r['text'][:500]}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"


@tool
def summarize_text(text: str) -> str:
    """Create a concise summary of the provided text.

    Use this when you have a long piece of text and need to extract
    the key points. This uses simple extractive summarization.

    Args:
        text: The text to summarize.

    Returns:
        A condensed version highlighting key sentences.
    """
    # Simple extractive summarization: pick the most important sentences.
    # A production system would use an LLM for this, but for a tool
    # that the agent itself calls, we use a lightweight approach.
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

    if len(sentences) <= 5:
        return text  # Already short enough

    # Take first 2 sentences (usually intro/topic), middle, and last 2 (conclusion)
    selected = sentences[:2] + [sentences[len(sentences) // 2]] + sentences[-2:]
    return ". ".join(selected) + "."


# ── Tool Registry ──────────────────────────────────────────────────────
# All tools available to agents, collected in a list for easy registration.
# Phase 3 tools (research) + Phase 4 tools (utilities) combined here.

from agents.tool_library import UTILITY_TOOLS

ALL_TOOLS = [
    # Phase 3: Research tools
    web_search,
    scrape_webpage,
    knowledge_base_search,
    summarize_text,
    # Phase 4: Utility tools (calculator, code runner, datetime, file writer)
    *UTILITY_TOOLS,
]
