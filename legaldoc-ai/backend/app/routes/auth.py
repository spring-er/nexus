"""Auth routes — signup, login, refresh, me."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from supabase import create_client

from app.config import settings
from app.dependencies import get_current_user
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Supabase helper ───────────────────────────────────


def _get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ── Request / response schemas ────────────────────────


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    created_at: str


class AuthResponse(BaseModel):
    user: UserResponse
    tokens: TokenResponse


# ── Helpers ───────────────────────────────────────────


def _user_response(row: dict) -> UserResponse:
    """Build a ``UserResponse`` from a Supabase row dict."""
    return UserResponse(
        id=row["id"],
        email=row["email"],
        full_name=row["full_name"],
        created_at=row["created_at"],
    )


def _token_pair(user_id: str) -> TokenResponse:
    """Generate a fresh access + refresh token pair."""
    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
    )


# ── POST /signup ──────────────────────────────────────


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest):
    """Register a new user account.

    Returns the created user profile and a token pair.
    """
    client = _get_supabase()

    # Check if email already exists.
    existing = (
        client.table("users")
        .select("id")
        .eq("email", body.email)
        .maybe_single()
        .execute()
    )
    if existing.data is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    now = datetime.now(timezone.utc).isoformat()

    row = {
        "email": body.email,
        "password_hash": hash_password(body.password),
        "full_name": body.full_name,
        "document_count": 0,
        "created_at": now,
        "updated_at": now,
    }

    result = client.table("users").insert(row).execute()
    user = result.data[0]

    logger.info("New user registered: %s (%s)", user["id"], body.email)

    return AuthResponse(
        user=_user_response(user),
        tokens=_token_pair(user["id"]),
    )


# ── POST /login ──────────────────────────────────────


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    """Authenticate with email and password.

    Returns the user profile and a fresh token pair.
    """
    client = _get_supabase()

    result = (
        client.table("users")
        .select("*")
        .eq("email", body.email)
        .maybe_single()
        .execute()
    )

    if result.data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = result.data

    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    logger.info("User logged in: %s", user["id"])

    return AuthResponse(
        user=_user_response(user),
        tokens=_token_pair(user["id"]),
    )


# ── POST /refresh ────────────────────────────────────


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest):
    """Exchange a valid refresh token for a new token pair.

    The refresh token must be of type ``"refresh"`` and not expired.
    """
    try:
        payload = decode_token(body.refresh_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — expected refresh token",
        )

    user_id: str = payload["sub"]

    # Verify the user still exists.
    client = _get_supabase()
    result = (
        client.table("users")
        .select("id")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    if result.data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return _token_pair(user_id)


# ── GET /me ──────────────────────────────────────────


@router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return _user_response(current_user)
