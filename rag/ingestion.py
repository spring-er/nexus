"""
rag/ingestion.py — Document Ingestion Pipeline

This module handles the first step of RAG: taking raw documents and
converting them into small, searchable text chunks.

THE RAG PIPELINE:
    [Document] → INGEST → [Chunks] → Embed → [Vectors] → Store → [Vector DB]
                 ^^^^^^
                 WE ARE HERE

WHAT INGESTION DOES:
    1. Reads the document (PDF, TXT, MD, or web URL)
    2. Extracts the text content
    3. Splits the text into overlapping chunks
    4. Returns the chunks with metadata (source filename, page number, etc.)

WHY OVERLAPPING CHUNKS?
    If we split a document at exactly every 512 tokens with no overlap,
    a sentence that spans the boundary gets cut in half. Overlapping by
    ~50 tokens means the boundary region appears in two chunks, so
    nothing is lost. Think of it like overlapping tiles on a roof.

SUPPORTED FORMATS:
    - PDF: Uses pypdf (via LlamaIndex) to extract text from each page
    - TXT/MD: Read directly as text
    - Web URLs: Fetch the page, extract text with BeautifulSoup
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from llama_index.readers.file import (
    PDFReader,
)


# ── Data Classes ────────────────────────────────────────────────────────


@dataclass
class ChunkMetadata:
    """
    Metadata attached to each chunk for tracking and citation.

    When the RAG engine retrieves a chunk to answer a question,
    this metadata lets us tell the user WHERE the answer came from.

    Attributes:
        source: Original filename or URL.
        source_type: "pdf", "txt", "md", "url", or "youtube".
        chunk_index: Position of this chunk within the document.
        total_chunks: Total number of chunks from this document.
        page_number: PDF page number (if applicable).
        title: Document title or YouTube video title.
    """

    source: str
    source_type: str
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: int | None = None
    title: str = ""


@dataclass
class TextChunk:
    """
    A single chunk of text ready for embedding.

    This is the unit that gets embedded and stored in the vector database.
    Each chunk carries its text content plus metadata for citation.

    Attributes:
        text: The actual text content of this chunk.
        metadata: Where this chunk came from (for citations).
    """

    text: str
    metadata: ChunkMetadata


# ── Ingestion Functions ─────────────────────────────────────────────────

# The SentenceSplitter is LlamaIndex's text chunking tool.
# It splits on sentence boundaries (not mid-word) and adds overlap.
#
# Parameters:
#   chunk_size=512: Each chunk is ~512 tokens (~380 words).
#                   This is a sweet spot: big enough for context, small
#                   enough for precise retrieval.
#   chunk_overlap=50: 50 tokens of overlap between adjacent chunks.
#                     Prevents information loss at chunk boundaries.
DEFAULT_SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def ingest_text_file(file_path: str) -> list[TextChunk]:
    """
    Ingest a plain text or markdown file.

    This is the simplest case: read the file, split into chunks.

    Args:
        file_path: Path to a .txt or .md file.

    Returns:
        List of TextChunk objects ready for embedding.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    source_type = path.suffix.lstrip(".").lower()  # "txt" or "md"

    return _split_text_into_chunks(
        text=text,
        source=path.name,
        source_type=source_type,
        title=path.stem,  # Filename without extension
    )


def ingest_pdf(file_path: str) -> list[TextChunk]:
    """
    Ingest a PDF file using LlamaIndex's PDF reader.

    PDFs are complex — they can contain images, tables, columns, etc.
    LlamaIndex's PDFReader (powered by pypdf) handles the text extraction.
    It returns one Document per page, which we then chunk.

    Args:
        file_path: Path to a .pdf file.

    Returns:
        List of TextChunk objects with page numbers in metadata.
    """
    reader = PDFReader()

    # LlamaIndex returns a list of Document objects, one per page.
    documents: list[Document] = reader.load_data(file=Path(file_path))

    all_chunks: list[TextChunk] = []
    filename = Path(file_path).name

    for page_num, doc in enumerate(documents, start=1):
        # Skip empty pages
        if not doc.text.strip():
            continue

        page_chunks = _split_text_into_chunks(
            text=doc.text,
            source=filename,
            source_type="pdf",
            title=Path(file_path).stem,
            page_number=page_num,
        )
        all_chunks.extend(page_chunks)

    # Update total_chunks now that we know the final count
    for i, chunk in enumerate(all_chunks):
        chunk.metadata.chunk_index = i
        chunk.metadata.total_chunks = len(all_chunks)

    return all_chunks


def ingest_url(url: str) -> list[TextChunk]:
    """
    Ingest a web page by fetching it and extracting text content.

    We use httpx to fetch the page and BeautifulSoup to parse the HTML
    and extract readable text. This approach is simpler and more reliable
    than the llama-index web reader (which has many dependencies).

    TRADEOFF: httpx+BS4 vs llama-index-readers-web
        - httpx+BS4: Fewer dependencies, more control, handles most pages
        - llama-index web reader: More features but heavy dependencies
        We chose simplicity since most knowledge base URLs are articles/docs.

    Args:
        url: The web URL to ingest.

    Returns:
        List of TextChunk objects from the page content.
    """
    # Fetch the web page
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    # Parse HTML and extract text
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements (they're not content)
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Get the page title
    title = soup.title.string if soup.title else url

    # Extract text, collapsing whitespace
    text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        raise ValueError(f"No readable text found at {url}")

    return _split_text_into_chunks(
        text=text,
        source=url,
        source_type="url",
        title=title or url,
    )


def ingest_file(file_path: str) -> list[TextChunk]:
    """
    Auto-detect file type and ingest accordingly.

    This is the main entry point for file ingestion. It looks at the
    file extension to decide which specific ingestion function to call.

    Args:
        file_path: Path to any supported file (PDF, TXT, MD).

    Returns:
        List of TextChunk objects.

    Raises:
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return ingest_pdf(file_path)
    elif suffix in (".txt", ".md", ".markdown"):
        return ingest_text_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: .pdf, .txt, .md"
        )


def ingest_uploaded_file(
    file_path: str, original_filename: str
) -> list[TextChunk]:
    """
    Ingest a file uploaded through Gradio.

    Gradio saves uploaded files to a temp directory with a random name.
    We need the original filename to detect the file type, and we copy
    the file to our documents directory for persistence.

    Args:
        file_path: Temporary path where Gradio saved the uploaded file.
        original_filename: The original name of the uploaded file.

    Returns:
        List of TextChunk objects.
    """
    # Copy to our documents directory so it persists
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest_path = docs_dir / original_filename

    # If a file with the same name exists, make it unique
    if dest_path.exists():
        stem = dest_path.stem
        suffix = dest_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = docs_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    # Copy the file
    import shutil
    shutil.copy2(file_path, dest_path)

    return ingest_file(str(dest_path))


# ── Private Helpers ─────────────────────────────────────────────────────


def _split_text_into_chunks(
    text: str,
    source: str,
    source_type: str,
    title: str = "",
    page_number: int | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks using LlamaIndex's SentenceSplitter.

    This is the core chunking logic used by all ingestion functions.

    Args:
        text: The full text to split.
        source: Source identifier (filename or URL) for metadata.
        source_type: Type of source ("pdf", "txt", "md", "url", "youtube").
        title: Human-readable title for the document.
        page_number: PDF page number (None for non-PDF sources).

    Returns:
        List of TextChunk objects with metadata.
    """
    # LlamaIndex's SentenceSplitter needs a Document object
    doc = Document(text=text)

    # Split into nodes (LlamaIndex's term for chunks)
    nodes = DEFAULT_SPLITTER.get_nodes_from_documents([doc])

    chunks = []
    for i, node in enumerate(nodes):
        chunk = TextChunk(
            text=node.text,
            metadata=ChunkMetadata(
                source=source,
                source_type=source_type,
                chunk_index=i,
                total_chunks=len(nodes),
                page_number=page_number,
                title=title,
            ),
        )
        chunks.append(chunk)

    return chunks
