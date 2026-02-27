"""Text extraction service for PDF, DOCX, and image files."""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when text extraction fails."""


# ── PDF extraction (PyMuPDF / fitz) ───────────────────


def extract_text_from_pdf(file_bytes: bytes) -> dict:
    """Extract text from a PDF file using PyMuPDF.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        {"text": str, "page_count": int, "pages": list[dict]}
    """
    import fitz  # PyMuPDF

    if not file_bytes:
        raise ExtractionError("Cannot extract text from an empty file")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(f"Failed to open PDF: {exc}") from exc

    pages: list[dict] = []
    full_text_parts: list[str] = []

    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            page_text = page.get_text("text")
            pages.append({
                "page_number": page_number + 1,
                "text": page_text,
                "char_count": len(page_text),
            })
            full_text_parts.append(page_text)
    finally:
        doc.close()

    return {
        "text": "\n".join(full_text_parts),
        "page_count": len(pages),
        "pages": pages,
    }


# ── DOCX extraction (python-docx) ────────────────────


def extract_text_from_docx(file_bytes: bytes) -> dict:
    """Extract text from a DOCX file using python-docx.

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        {"text": str, "page_count": int, "pages": list[dict]}
    """
    from docx import Document as DocxDocument
    from docx.opc.exceptions import PackageNotFoundError

    if not file_bytes:
        raise ExtractionError("Cannot extract text from an empty file")

    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
    except PackageNotFoundError as exc:
        raise ExtractionError(f"Invalid or corrupted DOCX file: {exc}") from exc
    except Exception as exc:
        raise ExtractionError(f"Failed to open DOCX: {exc}") from exc

    paragraphs: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    # DOCX files don't have a native page concept accessible via python-docx.
    # Treat the entire document as a single logical page.
    pages = [
        {
            "page_number": 1,
            "text": full_text,
            "char_count": len(full_text),
        }
    ]

    return {
        "text": full_text,
        "page_count": 1,
        "pages": pages,
    }


# ── Image extraction (pytesseract + Pillow) ──────────


def extract_text_from_image(file_bytes: bytes) -> dict:
    """Extract text from an image file using Tesseract OCR.

    Args:
        file_bytes: Raw bytes of an image (PNG, JPEG, TIFF, etc.).

    Returns:
        {"text": str, "page_count": int, "pages": list[dict]}
    """
    import pytesseract
    from PIL import Image, UnidentifiedImageError

    if not file_bytes:
        raise ExtractionError("Cannot extract text from an empty file")

    try:
        image = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ExtractionError(f"Unrecognized image format: {exc}") from exc
    except Exception as exc:
        raise ExtractionError(f"Failed to open image: {exc}") from exc

    try:
        text: str = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError(
            "Tesseract OCR is not installed or not found in PATH"
        ) from exc
    except Exception as exc:
        raise ExtractionError(f"OCR processing failed: {exc}") from exc

    text = text.strip()

    pages = [
        {
            "page_number": 1,
            "text": text,
            "char_count": len(text),
        }
    ]

    return {
        "text": text,
        "page_count": 1,
        "pages": pages,
    }


# ── Router ────────────────────────────────────────────

_SUPPORTED_TYPES: dict[str, str] = {
    "pdf": "pdf",
    "docx": "docx",
    "doc": "docx",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tiff": "image",
    "tif": "image",
    "bmp": "image",
    "webp": "image",
}

_EXTRACTORS = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "image": extract_text_from_image,
}


def extract_text(file_bytes: bytes, file_type: str) -> dict:
    """Route extraction to the correct handler based on file type.

    Args:
        file_bytes: Raw bytes of the file.
        file_type: File extension (e.g. "pdf", "docx", "png").
                   Case-insensitive; leading dots are stripped.

    Returns:
        {"text": str, "page_count": int, "pages": list[dict]}

    Raises:
        ExtractionError: On empty input, unsupported format, or extraction failure.
    """
    if not file_bytes:
        raise ExtractionError("Cannot extract text from an empty file")

    normalized = file_type.lower().lstrip(".")
    category = _SUPPORTED_TYPES.get(normalized)

    if category is None:
        supported = ", ".join(sorted(_SUPPORTED_TYPES.keys()))
        raise ExtractionError(
            f"Unsupported file type: '{normalized}'. "
            f"Supported types: {supported}"
        )

    extractor = _EXTRACTORS[category]

    logger.info("Extracting text from %s file (%d bytes)", normalized, len(file_bytes))

    try:
        result = extractor(file_bytes)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(
            f"Unexpected error extracting text from {normalized} file: {exc}"
        ) from exc

    logger.info(
        "Extraction complete: %d pages, %d characters",
        result["page_count"],
        len(result["text"]),
    )

    return result
