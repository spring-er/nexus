"""Document service — Supabase-backed CRUD and analysis storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from math import ceil
from typing import Any

from supabase import Client, create_client

from app.config import settings

logger = logging.getLogger(__name__)

# ── Supabase client ───────────────────────────────────


def _get_client() -> Client:
    """Create and return a Supabase client using the service role key."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ── Helpers ───────────────────────────────────────────


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── 1. create_document ────────────────────────────────


def create_document(
    user_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    storage_path: str,
) -> dict:
    """Insert a new document record.

    Args:
        user_id:      Owner's user ID.
        filename:     Original filename.
        file_type:    File extension (e.g. "pdf").
        file_size:    Size in bytes.
        storage_path: Supabase Storage path.

    Returns:
        The newly created document row as a dict.
    """
    client = _get_client()

    row = {
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "storage_path": storage_path,
        "status": "uploaded",
        "status_message": "File uploaded successfully",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    result = client.table("documents").insert(row).execute()
    return result.data[0]


# ── 2. get_document ───────────────────────────────────


def get_document(document_id: str, user_id: str) -> dict | None:
    """Fetch a single document, enforcing ownership.

    Args:
        document_id: Document UUID.
        user_id:     Owner's user ID (must match).

    Returns:
        The document row dict, or ``None`` if not found / not owned.
    """
    client = _get_client()

    result = (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return result.data


# ── 3. list_documents ─────────────────────────────────


def list_documents(
    user_id: str,
    page: int = 1,
    per_page: int = 20,
    filters: dict[str, Any] | None = None,
) -> dict:
    """List documents for a user with pagination and optional filters.

    Args:
        user_id:  Owner's user ID.
        page:     Page number (1-indexed).
        per_page: Results per page.
        filters:  Optional column-value pairs to further narrow results
                  (e.g. ``{"status": "completed", "file_type": "pdf"}``).

    Returns:
        ``{"documents": [...], "total": int, "page": int, "per_page": int,
        "total_pages": int}``
    """
    client = _get_client()

    # --- count ---
    count_query = (
        client.table("documents")
        .select("id", count="exact")
        .eq("user_id", user_id)
    )
    if filters:
        for col, val in filters.items():
            count_query = count_query.eq(col, val)
    count_result = count_query.execute()
    total = count_result.count or 0

    # --- paginated rows ---
    offset = (page - 1) * per_page
    data_query = (
        client.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .range(offset, offset + per_page - 1)
    )
    if filters:
        for col, val in filters.items():
            data_query = data_query.eq(col, val)
    data_result = data_query.execute()

    return {
        "documents": data_result.data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": ceil(total / per_page) if per_page else 0,
    }


# ── 4. update_document_status ─────────────────────────


def update_document_status(
    document_id: str,
    status: str,
    message: str | None = None,
) -> None:
    """Update a document's processing status.

    Args:
        document_id: Document UUID.
        status:      New status value (e.g. "processing", "completed", "failed").
        message:     Optional human-readable status message.
    """
    client = _get_client()

    payload: dict[str, Any] = {
        "status": status,
        "updated_at": _now_iso(),
    }
    if message is not None:
        payload["status_message"] = message

    client.table("documents").update(payload).eq("id", document_id).execute()


# ── 5. update_document_text ───────────────────────────


def update_document_text(
    document_id: str,
    text: str,
    page_count: int,
) -> None:
    """Store extracted text and page count on the document.

    Args:
        document_id: Document UUID.
        text:        Full extracted text.
        page_count:  Number of pages extracted.
    """
    client = _get_client()

    client.table("documents").update({
        "extracted_text": text,
        "page_count": page_count,
        "updated_at": _now_iso(),
    }).eq("id", document_id).execute()


# ── 6. store_extracted_data ───────────────────────────


def store_extracted_data(document_id: str, data: dict) -> None:
    """Store Pass-1 classification / extraction results.

    Saves the full JSON blob in ``extracted_metadata`` and promotes
    top-level fields (document_type, parties, etc.) when present.

    Args:
        document_id: Document UUID.
        data:        The dict returned by ``classify_document()``.
    """
    client = _get_client()

    payload: dict[str, Any] = {
        "extracted_metadata": json.dumps(data),
        "updated_at": _now_iso(),
    }

    if "document_type" in data:
        payload["document_type"] = data["document_type"]
    if "parties" in data:
        payload["parties"] = json.dumps(data["parties"])

    client.table("documents").update(payload).eq("id", document_id).execute()


# ── 7. store_clauses ─────────────────────────────────


def store_clauses(document_id: str, clause_data: dict) -> None:
    """Store Pass-2 clause analysis results.

    Each clause is stored as a row in the ``clauses`` table, and the full
    analysis blob (including risks, compliance, missing clauses) is saved on
    the ``documents`` row.

    Args:
        document_id: Document UUID.
        clause_data: The dict returned by ``analyze_clauses()``.
    """
    client = _get_client()

    # Persist the full analysis JSON on the document row.
    client.table("documents").update({
        "clause_analysis": json.dumps(clause_data),
        "updated_at": _now_iso(),
    }).eq("id", document_id).execute()

    # Insert individual clause rows for granular querying.
    clauses = clause_data.get("clauses", [])
    if clauses:
        rows = [
            {
                "document_id": document_id,
                "clause_title": c.get("clause_title", ""),
                "clause_text": c.get("clause_text", ""),
                "category": c.get("category", "Other"),
                "importance": c.get("importance", "medium"),
                "page_reference": c.get("page_reference", "N/A"),
                "created_at": _now_iso(),
            }
            for c in clauses
        ]
        client.table("clauses").insert(rows).execute()

    # Insert individual risk rows.
    risks = clause_data.get("risks", [])
    if risks:
        risk_rows = [
            {
                "document_id": document_id,
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "severity": r.get("severity", "medium"),
                "clause_reference": r.get("clause_reference", ""),
                "recommendation": r.get("recommendation", ""),
                "created_at": _now_iso(),
            }
            for r in risks
        ]
        client.table("risks").insert(risk_rows).execute()


# ── 8. store_summary ─────────────────────────────────


def store_summary(
    document_id: str,
    summary_data: dict,
    clause_data: dict,
) -> None:
    """Store Pass-3 executive summary results.

    Args:
        document_id:  Document UUID.
        summary_data: The dict returned by ``generate_summary()``.
        clause_data:  The clause analysis dict (for overall risk context).
    """
    client = _get_client()

    payload: dict[str, Any] = {
        "summary": json.dumps(summary_data),
        "updated_at": _now_iso(),
    }

    if "title" in summary_data:
        payload["title"] = summary_data["title"]
    if "overall_risk_rating" in summary_data:
        payload["overall_risk_rating"] = summary_data["overall_risk_rating"]
    if "executive_summary" in summary_data:
        payload["executive_summary"] = summary_data["executive_summary"]

    client.table("documents").update(payload).eq("id", document_id).execute()

    # Insert action items for easy querying.
    action_items = summary_data.get("action_items", [])
    if action_items:
        rows = [
            {
                "document_id": document_id,
                "action": a.get("action", ""),
                "priority": a.get("priority", "medium"),
                "deadline": a.get("deadline", "No deadline specified"),
                "created_at": _now_iso(),
            }
            for a in action_items
        ]
        client.table("action_items").insert(rows).execute()


# ── 9. finalize_document ─────────────────────────────


def finalize_document(
    document_id: str,
    document_type: str,
    processing_time: float,
) -> None:
    """Mark a document as fully processed.

    Args:
        document_id:     Document UUID.
        document_type:   Final classified type (e.g. "NDA").
        processing_time: Total wall-clock seconds for the pipeline.
    """
    client = _get_client()

    client.table("documents").update({
        "status": "completed",
        "status_message": "Analysis complete",
        "document_type": document_type,
        "processing_time": processing_time,
        "completed_at": _now_iso(),
        "updated_at": _now_iso(),
    }).eq("id", document_id).execute()


# ── 10. delete_document ──────────────────────────────


def delete_document(document_id: str, user_id: str) -> bool:
    """Delete a document and its related rows, enforcing ownership.

    Args:
        document_id: Document UUID.
        user_id:     Owner's user ID (must match).

    Returns:
        ``True`` if a matching document was deleted, ``False`` otherwise.
    """
    client = _get_client()

    # Verify ownership first.
    doc = get_document(document_id, user_id)
    if doc is None:
        return False

    # Delete child rows first (clauses, risks, action_items).
    client.table("action_items").delete().eq("document_id", document_id).execute()
    client.table("risks").delete().eq("document_id", document_id).execute()
    client.table("clauses").delete().eq("document_id", document_id).execute()

    # Delete the document row.
    client.table("documents").delete().eq("id", document_id).eq("user_id", user_id).execute()

    return True


# ── 11. search_documents ─────────────────────────────


def search_documents(user_id: str, query: str) -> list[dict]:
    """Full-text search across a user's documents.

    Searches the ``filename``, ``title``, and ``executive_summary`` columns
    using Supabase's ``ilike`` operator.

    Args:
        user_id: Owner's user ID.
        query:   Search term.

    Returns:
        List of matching document rows (newest first).
    """
    client = _get_client()

    pattern = f"%{query}%"

    result = (
        client.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .or_(
            f"filename.ilike.{pattern},"
            f"title.ilike.{pattern},"
            f"executive_summary.ilike.{pattern}"
        )
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


# ── 12. increment_user_doc_count ─────────────────────


def increment_user_doc_count(user_id: str) -> None:
    """Increment the ``document_count`` field on the user's profile.

    Uses a Supabase RPC call to atomically increment the counter.

    Args:
        user_id: The user whose count should be bumped.
    """
    client = _get_client()

    client.rpc("increment_document_count", {"p_user_id": user_id}).execute()
