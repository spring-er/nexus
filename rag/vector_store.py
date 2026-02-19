"""
rag/vector_store.py — LanceDB Vector Store Operations

This module handles storing and searching embedding vectors using LanceDB,
a local vector database that saves data as files on disk.

WHAT IS A VECTOR DATABASE?
    A regular database finds rows by exact matching (WHERE name = "Alice").
    A vector database finds rows by SIMILARITY — "find the 5 vectors
    closest to this query vector." This is how RAG retrieval works:
    embed the question, search for similar document chunks.

WHY LANCEDB?
    - Runs locally: No server, no cloud account, no cost
    - Persistent: Data survives restarts (stored in data/vectordb/)
    - Fast: Uses Apache Arrow format for efficient vector search
    - Simple API: Just connect, create table, search

HOW VECTOR SEARCH WORKS:
    1. We store chunks as rows: {text, vector, source, source_type, ...}
    2. When querying, we embed the question into a vector
    3. LanceDB computes cosine similarity between the query vector and
       every stored vector
    4. Returns the top-k most similar chunks (default k=5)
    5. These chunks become the "context" for the LLM to answer from

SCHEMA:
    Each row in our LanceDB table has:
        - text (str): The chunk's text content
        - vector (list[float]): 384-dimensional embedding
        - source (str): Filename or URL
        - source_type (str): "pdf", "txt", "md", "url", "youtube"
        - title (str): Document/video title
        - chunk_index (int): Position within the document
        - doc_id (str): Unique ID for the source document (for deletion)
"""

import hashlib
from pathlib import Path

import lancedb

from rag.embeddings import embed_texts, embed_query
from rag.ingestion import TextChunk


# ── LanceDB Configuration ──────────────────────────────────────────────

# Where the vector database files are stored
VECTORDB_PATH = "data/vectordb"

# Name of the table within the database
TABLE_NAME = "knowledge_base"

# How many results to return by default when searching
DEFAULT_TOP_K = 5


def _get_db() -> lancedb.DBConnection:
    """
    Connect to (or create) the LanceDB database.

    LanceDB stores data in a directory. If the directory doesn't exist,
    it creates it. If it does exist, it reconnects to the existing data.

    Returns:
        A LanceDB connection object.
    """
    Path(VECTORDB_PATH).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(VECTORDB_PATH)


def _table_exists(db: lancedb.DBConnection) -> bool:
    """Check if our knowledge base table exists in the database."""
    tables = db.list_tables()
    # list_tables() may return a list of strings or a paged result object
    if isinstance(tables, list):
        return TABLE_NAME in tables
    # Paged result — iterate to check
    return any(t == TABLE_NAME for t in tables)


def _generate_doc_id(source: str) -> str:
    """
    Generate a unique document ID from the source path/URL.

    We use a hash of the source to create a consistent, unique ID.
    This lets us delete all chunks from a specific document later.

    Args:
        source: The source filename or URL.

    Returns:
        A hex string hash (first 16 chars of SHA-256).
    """
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def add_chunks(chunks: list[TextChunk]) -> int:
    """
    Embed text chunks and add them to the vector store.

    This is the main function for adding documents to the knowledge base.
    It:
        1. Embeds all chunks into vectors (batch operation for speed)
        2. Builds rows with text + vector + metadata
        3. Adds them to the LanceDB table

    Args:
        chunks: List of TextChunk objects from the ingestion pipeline.

    Returns:
        Number of chunks added.

    Raises:
        ValueError: If the chunks list is empty.
    """
    if not chunks:
        raise ValueError("No chunks to add")

    # Step 1: Embed all chunk texts at once (batching is faster)
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)

    # Step 2: Build the data rows
    # Each row has: text, vector, and all metadata fields
    data = []
    for chunk, vector in zip(chunks, vectors):
        doc_id = _generate_doc_id(chunk.metadata.source)
        data.append({
            "text": chunk.text,
            "vector": vector,
            "source": chunk.metadata.source,
            "source_type": chunk.metadata.source_type,
            "title": chunk.metadata.title,
            "chunk_index": chunk.metadata.chunk_index,
            "doc_id": doc_id,
        })

    # Step 3: Add to LanceDB
    db = _get_db()

    if _table_exists(db):
        # Table exists: append new data
        table = db.open_table(TABLE_NAME)
        table.add(data)
    else:
        # Table doesn't exist: create it with the data
        # LanceDB infers the schema from the first batch of data
        db.create_table(TABLE_NAME, data)

    return len(data)


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search the knowledge base for chunks relevant to a query.

    This is the core of RAG retrieval:
        1. Embed the query into a vector
        2. Find the top_k most similar chunks in the database
        3. Return them with their text and metadata

    Args:
        query: The user's question or search text.
        top_k: Number of results to return (default 5).

    Returns:
        List of dicts, each with keys:
            - text: The chunk's text content
            - source: Where the chunk came from
            - source_type: Type of source
            - title: Document title
            - score: Similarity score (lower = more similar in LanceDB)
        Returns empty list if the knowledge base is empty.
    """
    db = _get_db()

    if not _table_exists(db):
        return []

    # Step 1: Embed the query
    query_vector = embed_query(query)

    # Step 2: Search LanceDB
    # LanceDB uses L2 (Euclidean) distance by default.
    # Lower distance = more similar.
    table = db.open_table(TABLE_NAME)
    results = (
        table.search(query_vector)
        .limit(top_k)
        .to_list()
    )

    # Step 3: Format results
    formatted = []
    for row in results:
        formatted.append({
            "text": row["text"],
            "source": row["source"],
            "source_type": row["source_type"],
            "title": row["title"],
            "score": row.get("_distance", 0.0),
        })

    return formatted


def list_documents() -> list[dict]:
    """
    List all unique documents in the knowledge base.

    Used by the UI to show what's been indexed. Groups chunks by
    doc_id and returns one entry per document.

    Returns:
        List of dicts with: source, source_type, title, chunk_count, doc_id
    """
    db = _get_db()

    if not _table_exists(db):
        return []

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    if df.empty:
        return []

    # Group by doc_id to get unique documents
    docs = []
    for doc_id, group in df.groupby("doc_id"):
        docs.append({
            "source": group["source"].iloc[0],
            "source_type": group["source_type"].iloc[0],
            "title": group["title"].iloc[0],
            "chunk_count": len(group),
            "doc_id": doc_id,
        })

    return docs


def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to a specific document.

    Args:
        doc_id: The unique document ID (from list_documents).

    Returns:
        Number of chunks deleted.
    """
    db = _get_db()

    if not _table_exists(db):
        return 0

    table = db.open_table(TABLE_NAME)

    # Count chunks before deletion
    df = table.to_pandas()
    count_before = len(df[df["doc_id"] == doc_id])

    if count_before == 0:
        return 0

    # LanceDB delete uses a SQL-like filter string
    table.delete(f"doc_id = '{doc_id}'")

    return count_before


def get_stats() -> dict:
    """
    Get knowledge base statistics.

    Returns:
        Dict with: total_chunks, total_documents, sources_by_type
    """
    db = _get_db()

    if not _table_exists(db):
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "sources_by_type": {},
        }

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    if df.empty:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "sources_by_type": {},
        }

    return {
        "total_chunks": len(df),
        "total_documents": df["doc_id"].nunique(),
        "sources_by_type": df["source_type"].value_counts().to_dict(),
    }
