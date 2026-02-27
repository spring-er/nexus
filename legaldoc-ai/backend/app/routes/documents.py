"""Document routes — upload, list, detail, status, compare, delete."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.services import ai_engine, document_service, storage_service
from app.services.processing_pipeline import process_document

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_FILE_TYPES = {"pdf", "docx", "doc", "png", "jpg", "jpeg", "tiff", "tif"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ── Response schemas ──────────────────────────────────


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    status_message: str
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[dict]
    total: int
    page: int
    per_page: int
    total_pages: int


class DocumentStatusResponse(BaseModel):
    id: str
    status: str
    status_message: str | None = None
    updated_at: str | None = None


# ── POST /upload ──────────────────────────────────────


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a document and start background processing.

    Accepts PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF files up to 50 MB.
    Returns immediately with status ``"uploaded"``; processing runs in
    the background.
    """
    user_id: str = current_user["id"]

    # ── Validate file type ────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{extension}. Allowed: {', '.join(sorted(ALLOWED_FILE_TYPES))}",
        )

    # ── Read & validate size ──────────────────────────
    file_bytes = file.file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({file_size} bytes). Maximum is {MAX_FILE_SIZE} bytes (50 MB)",
        )

    # ── Upload to storage ─────────────────────────────
    document_id = str(uuid.uuid4())

    try:
        storage_path = storage_service.upload_file(
            file_bytes, user_id, document_id, file.filename,
        )
    except Exception as exc:
        logger.exception("Storage upload failed for document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload file to storage: {exc}",
        )

    # ── Create DB record ──────────────────────────────
    try:
        doc = document_service.create_document(
            user_id=user_id,
            filename=file.filename,
            file_type=extension,
            file_size=file_size,
            storage_path=storage_path,
        )
    except Exception as exc:
        logger.exception("DB insert failed for document %s", document_id)
        # Best-effort cleanup of orphaned storage file.
        storage_service.delete_file(storage_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document record: {exc}",
        )

    # ── Kick off background processing ────────────────
    background_tasks.add_task(process_document, doc["id"], user_id)

    logger.info("Document %s uploaded by user %s — processing queued", doc["id"], user_id)

    return DocumentUploadResponse(
        id=doc["id"],
        filename=doc["filename"],
        file_type=doc["file_type"],
        file_size=doc["file_size"],
        status=doc["status"],
        status_message=doc.get("status_message", ""),
        created_at=doc["created_at"],
    )


# ── GET / (list) ──────────────────────────────────────


@router.get("", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    document_status: str | None = Query(None, alias="status"),
    file_type: str | None = Query(None),
    search: str | None = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """List the authenticated user's documents with pagination and filters."""
    user_id: str = current_user["id"]

    # Full-text search shortcut.
    if search:
        documents = document_service.search_documents(user_id, search)
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
            page=1,
            per_page=len(documents) or per_page,
            total_pages=1 if documents else 0,
        )

    filters: dict[str, str] = {}
    if document_status:
        filters["status"] = document_status
    if file_type:
        filters["file_type"] = file_type

    result = document_service.list_documents(
        user_id, page=page, per_page=per_page, filters=filters or None,
    )
    return DocumentListResponse(**result)


# ── GET /{id} (detail) ───────────────────────────────


@router.get("/{document_id}")
def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Retrieve full document detail including extraction results."""
    doc = document_service.get_document(document_id, current_user["id"])
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Parse JSON blobs so the response is a proper nested object.
    for field in ("extracted_metadata", "clause_analysis", "summary", "parties"):
        val = doc.get(field)
        if isinstance(val, str):
            try:
                doc[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass

    return doc


# ── GET /{id}/status (lightweight poll) ──────────────


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Lightweight status check for polling during processing."""
    doc = document_service.get_document(document_id, current_user["id"])
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return DocumentStatusResponse(
        id=doc["id"],
        status=doc["status"],
        status_message=doc.get("status_message"),
        updated_at=doc.get("updated_at"),
    )


# ── GET /{id}/compare/{other_id} ─────────────────────


@router.get("/{document_id}/compare/{other_id}")
def compare_documents(
    document_id: str,
    other_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Compare two documents owned by the current user.

    Both documents must be in ``"completed"`` status.
    """
    user_id: str = current_user["id"]

    doc_a = document_service.get_document(document_id, user_id)
    if doc_a is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    doc_b = document_service.get_document(other_id, user_id)
    if doc_b is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {other_id} not found",
        )

    if doc_a.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document {document_id} has not finished processing (status: {doc_a.get('status')})",
        )

    if doc_b.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document {other_id} has not finished processing (status: {doc_b.get('status')})",
        )

    try:
        comparison = ai_engine.compare_documents(doc_a, doc_b)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI comparison failed: {exc}",
        )

    return comparison


# ── DELETE /{id} ──────────────────────────────────────


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a document and its associated data and storage file."""
    user_id: str = current_user["id"]

    doc = document_service.get_document(document_id, user_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Delete from storage (best-effort).
    storage_path = doc.get("storage_path")
    if storage_path:
        storage_service.delete_file(storage_path)

    # Delete DB rows (cascading: action_items → risks → clauses → document).
    deleted = document_service.delete_document(document_id, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document record",
        )

    return {"message": "Document deleted successfully", "id": document_id}
