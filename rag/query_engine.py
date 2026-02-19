"""
rag/query_engine.py — RAG Query Engine (Retrieve + Generate)

This module ties together the full RAG pipeline:
    Retrieve relevant chunks → Build a context-augmented prompt → Generate answer

THIS IS WHERE RAG "HAPPENS":
    1. User asks: "What did the report say about Q3 revenue?"
    2. We search the vector store for chunks about Q3 revenue
    3. We build a prompt: "Given these documents: [chunks], answer: [question]"
    4. The LLM reads the chunks and answers from them
    5. We return the answer along with source citations

WHY NOT JUST SEND EVERYTHING?
    If you have 100 documents, you can't send them all to the LLM — it
    would exceed the context window and be very slow/expensive. RAG solves
    this by only sending the RELEVANT chunks (typically 3-5). The vector
    search ensures we pick the right ones.

SOURCE CITATIONS:
    One of the biggest problems with LLMs is that they can make things up
    (hallucinate). By showing WHICH documents the answer came from, users
    can verify the information. This builds trust and makes the tool useful
    for real research.
"""

from rag.vector_store import search as vector_search


def build_rag_context(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """
    Search the knowledge base and build a context string for the LLM.

    This function:
        1. Searches for relevant chunks
        2. Formats them into a readable context block
        3. Returns both the context string and the source metadata

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve (default 5).

    Returns:
        Tuple of:
            - context_str: Formatted context to inject into the prompt.
                           Empty string if no results found.
            - sources: List of source metadata dicts for citations.
    """
    # Step 1: Search for relevant chunks
    results = vector_search(query, top_k=top_k)

    if not results:
        return "", []

    # Step 2: Format chunks into a context block
    # We number each chunk and include its source for citation
    context_parts = []
    sources = []
    seen_sources = set()

    for i, result in enumerate(results, 1):
        # Build the context entry
        context_parts.append(
            f"[Document {i}] (Source: {result['title']})\n"
            f"{result['text']}\n"
        )

        # Track unique sources for citations
        source_key = result["source"]
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": result["source"],
                "source_type": result["source_type"],
                "title": result["title"],
            })

    context_str = "\n---\n".join(context_parts)

    return context_str, sources


def format_rag_prompt(query: str, context: str) -> str:
    """
    Build the RAG-enhanced user message.

    Instead of sending just the user's question, we prepend the relevant
    document context. The LLM is instructed to answer FROM this context.

    Args:
        query: The user's original question.
        context: The formatted context string from build_rag_context().

    Returns:
        A formatted prompt string that includes context + question.
    """
    if not context:
        # No knowledge base results — just return the original query
        return query

    return (
        "Use the following documents to answer the question. "
        "If the documents don't contain relevant information, say so "
        "and answer from your general knowledge instead.\n\n"
        "## Relevant Documents\n\n"
        f"{context}\n\n"
        "## Question\n\n"
        f"{query}"
    )


def format_citations(sources: list[dict]) -> str:
    """
    Format source citations as a markdown block for display.

    This creates a "Sources" section that appears below the AI's answer,
    showing which documents were used to generate the response.

    Args:
        sources: List of source metadata dicts from build_rag_context().

    Returns:
        A markdown-formatted string listing the sources.
        Empty string if no sources.
    """
    if not sources:
        return ""

    # Map source types to emoji-like labels
    type_labels = {
        "pdf": "PDF",
        "txt": "TXT",
        "md": "MD",
        "url": "Web",
        "youtube": "YouTube",
    }

    lines = ["\n\n---\n**Sources:**"]
    for source in sources:
        label = type_labels.get(source["source_type"], "Doc")
        lines.append(f"- [{label}] {source['title']} (`{source['source']}`)")

    return "\n".join(lines)
