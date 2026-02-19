"""
rag/embeddings.py — Embedding Generation for Nexus AI

This module converts text into vector embeddings using a HuggingFace
model that runs locally on your CPU. No API key or internet needed
(after the model is downloaded the first time).

WHAT ARE EMBEDDINGS?
    An embedding converts text into a list of numbers (a "vector") that
    captures its meaning. Similar texts produce similar vectors.

    Example:
        "How to fix a car" → [0.12, -0.45, 0.78, ...]
        "Auto repair guide" → [0.11, -0.44, 0.79, ...]  ← very similar!
        "Recipe for cookies" → [-0.67, 0.23, -0.12, ...]  ← very different

    By comparing vectors with cosine similarity, we can find the most
    relevant chunks for any question — even if the words don't match.

MODEL CHOICE: all-MiniLM-L6-v2
    - 80MB download (cached after first use)
    - 384-dimensional vectors
    - Runs on CPU in milliseconds
    - Good balance of quality and speed
    - Open-source (Apache 2.0 license)

    Tradeoff: Larger models (e.g., all-mpnet-base-v2 at 420MB) produce
    better embeddings but are slower. MiniLM is the standard choice
    for learning projects and smaller knowledge bases.

WHY LOCAL EMBEDDINGS?
    We COULD use OpenAI's embedding API or HuggingFace's Inference API,
    but running locally means:
    - Zero cost (no API calls)
    - Zero latency from network
    - Works offline
    - No rate limits
    - Complete privacy (your documents never leave your machine)
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ── Embedding Model ────────────────────────────────────────────────────

# The model name on HuggingFace Hub. First time you run this,
# it downloads ~80MB. After that, it's cached locally.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton: We only want ONE instance of the embedding model in memory.
# Loading a model is expensive (~1-2 seconds), so we do it once and reuse.
_embed_model: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """
    Get the embedding model instance (singleton pattern).

    The first call loads the model (takes ~1-2 seconds). Subsequent
    calls return the cached instance immediately.

    WHY SINGLETON?
        The embedding model uses ~100MB of RAM. If we created a new
        instance for every embedding request, we'd waste memory and
        time reloading. The singleton ensures one model, reused everywhere.

    Returns:
        A HuggingFaceEmbedding model ready to embed text.
    """
    global _embed_model

    if _embed_model is None:
        # LlamaIndex's HuggingFaceEmbedding wraps sentence-transformers.
        # It handles downloading, caching, and efficient batch embedding.
        _embed_model = HuggingFaceEmbedding(
            model_name=DEFAULT_EMBED_MODEL,
        )

    return _embed_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    This is a convenience function that handles batching. Embedding
    many texts at once is faster than embedding them one by one because
    the model can process them in parallel on the CPU.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors. Each vector is a list of 384 floats.
        The order matches the input: texts[i] → result[i].
    """
    model = get_embed_model()

    # _get_text_embeddings handles batching internally
    embeddings = [model.get_text_embedding(text) for text in texts]

    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    In some embedding models, queries and documents are embedded
    differently (query embeddings are optimized for search). LlamaIndex
    handles this distinction automatically.

    Args:
        query: The search query to embed.

    Returns:
        A single embedding vector (list of 384 floats).
    """
    model = get_embed_model()
    return model.get_query_embedding(query)
