"""Document processing pipeline — orchestrates extraction + 3-pass AI analysis."""

from __future__ import annotations

import json
import logging
import time

from app.services import (
    ai_engine,
    document_service,
    storage_service,
)
from app.services.extraction_service import extract_text

logger = logging.getLogger(__name__)


def process_document(document_id: str, user_id: str) -> None:
    """Run the full document-processing pipeline.

    Flow
    ----
    1. Mark status → ``"processing"``
    2. Download file bytes from Supabase Storage
    3. Extract text (PDF / DOCX / image)
    4. **Pass 1** — Classify document & extract metadata
    5. **Pass 2** — Clause-level analysis & risk assessment
    6. **Pass 3** — Executive summary generation
    7. Finalize document (status → ``"completed"``)
    8. Increment the user's document count

    On any unhandled exception the document status is set to ``"error"``
    with the exception message.

    This function is synchronous and designed to be invoked via
    ``fastapi.BackgroundTasks`` so that the upload endpoint returns
    immediately.

    Args:
        document_id: UUID of the document row to process.
        user_id:     Owner's user ID (for ownership-scoped lookups).
    """
    start = time.time()

    try:
        # ── 1. Mark as processing ────────────────────────
        document_service.update_document_status(
            document_id, "processing", "Processing started",
        )

        # ── 2. Fetch document record & download file ─────
        doc = document_service.get_document(document_id, user_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found for user {user_id}")

        storage_path: str = doc["storage_path"]
        file_type: str = doc["file_type"]

        document_service.update_document_status(
            document_id, "processing", "Downloading file",
        )
        file_bytes = storage_service.download_file(storage_path)
        logger.info("Downloaded %d bytes for document %s", len(file_bytes), document_id)

        # ── 3. Extract text ──────────────────────────────
        document_service.update_document_status(
            document_id, "processing", "Extracting text",
        )
        extraction = extract_text(file_bytes, file_type)
        document_text: str = extraction["text"]
        page_count: int = extraction["page_count"]

        if not document_text.strip():
            raise ValueError("Text extraction produced no content — the file may be empty or image-only without OCR text")

        document_service.update_document_text(document_id, document_text, page_count)
        logger.info(
            "Extracted %d chars / %d pages for document %s",
            len(document_text), page_count, document_id,
        )

        # ── 4. Pass 1 — Classification & metadata ───────
        document_service.update_document_status(
            document_id, "processing", "Classifying document (Pass 1 of 3)",
        )
        classification = ai_engine.classify_document(document_text)
        document_service.store_extracted_data(document_id, classification)

        document_type: str = classification.get("document_type", "Other")
        parties_list = classification.get("parties", [])
        parties_str = ", ".join(
            f"{p.get('name', 'Unknown')} ({p.get('role', 'Unknown')})"
            for p in parties_list
        ) or "Not identified"

        logger.info(
            "Pass 1 complete for %s — type=%s, parties=%s",
            document_id, document_type, parties_str,
        )

        # ── 5. Pass 2 — Clause analysis & risks ─────────
        document_service.update_document_status(
            document_id, "processing", "Analyzing clauses (Pass 2 of 3)",
        )
        clause_analysis = ai_engine.analyze_clauses(document_text, document_type)
        document_service.store_clauses(document_id, clause_analysis)

        logger.info(
            "Pass 2 complete for %s — %d clauses, %d risks",
            document_id,
            len(clause_analysis.get("clauses", [])),
            len(clause_analysis.get("risks", [])),
        )

        # ── 6. Pass 3 — Executive summary ───────────────
        document_service.update_document_status(
            document_id, "processing", "Generating summary (Pass 3 of 3)",
        )
        summary = ai_engine.generate_summary(
            document_text, document_type, parties_str, clause_analysis,
        )
        document_service.store_summary(document_id, summary, clause_analysis)

        logger.info("Pass 3 complete for %s", document_id)

        # ── 7. Finalize ─────────────────────────────────
        processing_time = time.time() - start
        document_service.finalize_document(document_id, document_type, processing_time)

        logger.info(
            "Document %s processed successfully in %.1fs",
            document_id, processing_time,
        )

        # ── 8. Increment user document count ─────────────
        try:
            document_service.increment_user_doc_count(user_id)
        except Exception:
            logger.warning(
                "Failed to increment doc count for user %s — non-critical, continuing",
                user_id,
                exc_info=True,
            )

    except Exception:
        elapsed = time.time() - start
        logger.exception(
            "Pipeline failed for document %s after %.1fs", document_id, elapsed,
        )
        try:
            document_service.update_document_status(
                document_id, "error", "Processing failed — please retry or contact support",
            )
        except Exception:
            logger.exception(
                "Failed to set error status on document %s", document_id,
            )
