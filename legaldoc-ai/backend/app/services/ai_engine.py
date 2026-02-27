"""AI engine — multi-pass document analysis powered by Claude."""

from __future__ import annotations

import json
import logging
import time

import anthropic

from app.config import settings
from app.prompts.classify_and_extract import (
    SYSTEM_PROMPT_CLASSIFY,
    USER_PROMPT_CLASSIFY,
)
from app.prompts.clause_analysis import (
    SYSTEM_PROMPT_CLAUSES,
    USER_PROMPT_CLAUSES,
)
from app.prompts.executive_summary import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
)

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 150_000
RATE_LIMIT_DELAY_SECONDS = 1


# ── Low-level Claude call ─────────────────────────────


def call_claude(system_prompt: str, user_prompt: str) -> dict:
    """Send a request to the Anthropic Claude API and return parsed JSON.

    Uses temperature=0 and max_tokens=4096 for deterministic, complete output.
    If Claude returns invalid JSON the call is retried once.  On a second
    failure a ``ValueError`` is raised.

    Args:
        system_prompt: The system-level instruction.
        user_prompt:   The user-level message (document + task).

    Returns:
        Parsed JSON dictionary from Claude's response.

    Raises:
        ValueError:  If Claude returns invalid JSON after one retry.
        anthropic.APIError: On unrecoverable API errors.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    last_raw_text: str = ""

    for attempt in range(2):
        if attempt > 0:
            logger.warning("Retrying Claude call (attempt %d) — previous response was not valid JSON", attempt + 1)
            time.sleep(RATE_LIMIT_DELAY_SECONDS)

        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        last_raw_text = message.content[0].text

        try:
            return json.loads(last_raw_text)
        except json.JSONDecodeError:
            logger.warning("Claude returned invalid JSON (attempt %d): %.200s…", attempt + 1, last_raw_text)
            continue

    raise ValueError(
        f"Claude returned invalid JSON after 2 attempts. Last response: {last_raw_text[:500]}"
    )


# ── Helper ────────────────────────────────────────────


def _truncate(text: str) -> str:
    """Truncate document text to MAX_DOCUMENT_CHARS if necessary."""
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    logger.warning(
        "Document text truncated from %d to %d characters",
        len(text),
        MAX_DOCUMENT_CHARS,
    )
    return text[:MAX_DOCUMENT_CHARS]


# ── Pass 1: Classification ────────────────────────────


def classify_document(document_text: str) -> dict:
    """Classify a legal document and extract key metadata.

    Args:
        document_text: Full extracted text of the document.

    Returns:
        Dict with document_type, jurisdiction, parties, dates, etc.
    """
    user_prompt = USER_PROMPT_CLASSIFY.format(
        document_text=_truncate(document_text),
    )

    result = call_claude(SYSTEM_PROMPT_CLASSIFY, user_prompt)

    time.sleep(RATE_LIMIT_DELAY_SECONDS)
    return result


# ── Pass 2: Clause analysis ──────────────────────────


def analyze_clauses(document_text: str, document_type: str) -> dict:
    """Analyze clauses, risks, and compliance issues in a legal document.

    Args:
        document_text: Full extracted text of the document.
        document_type: The document type string from Pass 1
                       (e.g. "NDA", "MSA").

    Returns:
        Dict with clauses, risks, compliance_issues, missing_clauses.
    """
    user_prompt = USER_PROMPT_CLAUSES.format(
        document_text=_truncate(document_text),
        document_type=document_type,
    )

    result = call_claude(SYSTEM_PROMPT_CLAUSES, user_prompt)

    time.sleep(RATE_LIMIT_DELAY_SECONDS)
    return result


# ── Pass 3: Executive summary ────────────────────────


def generate_summary(
    document_text: str,
    document_type: str,
    parties: str,
    clause_analysis: dict,
) -> dict:
    """Generate an executive summary incorporating all prior analysis.

    Args:
        document_text:   Full extracted text of the document.
        document_type:   The document type string from Pass 1.
        parties:         Human-readable string listing the parties.
        clause_analysis: The full clause-analysis dict from Pass 2.

    Returns:
        Dict with executive_summary, key_terms, action_items,
        overall_risk_rating, and recommendation.
    """
    user_prompt = USER_PROMPT_SUMMARY.format(
        document_text=_truncate(document_text),
        document_type=document_type,
        parties=parties,
        clause_analysis=json.dumps(clause_analysis, indent=2),
    )

    result = call_claude(SYSTEM_PROMPT_SUMMARY, user_prompt)

    # No delay after the final pass — nothing follows.
    return result
