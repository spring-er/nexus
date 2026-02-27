"""Supabase Storage service for document file management."""

from __future__ import annotations

import logging

from supabase import Client, create_client

from app.config import settings

logger = logging.getLogger(__name__)


def _get_client() -> Client:
    """Create and return a Supabase client using the service role key."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def upload_file(
    file_bytes: bytes,
    user_id: str,
    document_id: str,
    filename: str,
) -> str:
    """Upload a file to Supabase Storage.

    Args:
        file_bytes:  Raw file content.
        user_id:     Owner's user ID (used as path prefix for isolation).
        document_id: Unique document ID.
        filename:    Original filename (preserved in the storage path).

    Returns:
        The storage path string: ``{user_id}/{document_id}/{filename}``.

    Raises:
        Exception: On Supabase storage upload failure.
    """
    storage_path = f"{user_id}/{document_id}/{filename}"
    bucket = settings.supabase_storage_bucket
    client = _get_client()

    logger.info("Uploading file to %s/%s", bucket, storage_path)

    client.storage.from_(bucket).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/octet-stream"},
    )

    logger.info("Upload complete: %s (%d bytes)", storage_path, len(file_bytes))
    return storage_path


def download_file(storage_path: str) -> bytes:
    """Download a file from Supabase Storage.

    Args:
        storage_path: The path returned by :func:`upload_file`.

    Returns:
        Raw file bytes.

    Raises:
        Exception: On Supabase storage download failure.
    """
    bucket = settings.supabase_storage_bucket
    client = _get_client()

    logger.info("Downloading file from %s/%s", bucket, storage_path)

    data = client.storage.from_(bucket).download(storage_path)
    return data


def delete_file(storage_path: str) -> bool:
    """Delete a file from Supabase Storage.

    Args:
        storage_path: The path returned by :func:`upload_file`.

    Returns:
        ``True`` if the file was deleted successfully, ``False`` otherwise.
    """
    bucket = settings.supabase_storage_bucket
    client = _get_client()

    logger.info("Deleting file %s/%s", bucket, storage_path)

    try:
        client.storage.from_(bucket).remove([storage_path])
        return True
    except Exception:
        logger.exception("Failed to delete file: %s", storage_path)
        return False
