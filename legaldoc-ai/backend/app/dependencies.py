"""FastAPI dependencies — auth, database clients, etc."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client

from app.config import settings
from app.services.auth_service import decode_token

_bearer_scheme = HTTPBearer()


def _get_supabase():
    """Return a Supabase client using the service role key."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """Validate the Authorization Bearer token and return the user row.

    Steps:
        1. Extract the token from the ``Authorization: Bearer <token>`` header.
        2. Decode and verify the JWT (signature + expiry).
        3. Ensure the token is an *access* token (not a refresh token).
        4. Query the ``users`` table in Supabase by the token's ``sub`` claim.
        5. Return the full user row dict.

    Raises:
        HTTPException 401: On missing/invalid/expired token or unknown user.
    """
    token = credentials.credentials

    # Decode JWT
    try:
        payload = decode_token(token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Must be an access token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload["sub"]

    # Look up user in DB
    client = _get_supabase()
    result = (
        client.table("users")
        .select("*")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )

    if result.data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return result.data
