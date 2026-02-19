"""
monitoring/tracker.py — Usage Tracking & Analytics for Nexus AI

This module records every LLM interaction: tokens used, cost, latency,
model, and whether it succeeded or failed. Data is stored locally in
a JSON-lines file (one JSON object per line).

WHY MONITOR LLM USAGE?
    1. Cost control: LLM API calls cost money. Without tracking, costs
       can spiral. Knowing which models use the most tokens lets you
       optimize.
    2. Performance: If a model is slow, you want to know. Response time
       tracking helps you choose the fastest provider for each task.
    3. Debugging: When something goes wrong, logs tell you what happened.
    4. Portfolio: "I built monitoring that tracks X interactions with
       Y% success rate" is impressive on a resume.

DATA FORMAT — JSON Lines (.jsonl):
    Each line is a complete JSON object. This format is:
    - Append-only (fast writes, no need to parse the whole file)
    - Easy to process line-by-line (even for huge files)
    - Standard format used by OpenAI, LangSmith, etc.

    Example line:
    {"timestamp": "2024-01-15T10:30:00", "model": "llama-3.1-8b",
     "input_tokens": 150, "output_tokens": 200, "cost": 0.0001, ...}

COST ESTIMATION:
    We estimate costs based on known pricing. This isn't exact (prices
    change, and free tiers don't actually charge), but it gives users
    a sense of what they'd pay at scale. Many free-tier models show
    $0.00 cost, which is correct — they ARE free.

THREAD SAFETY:
    Multiple Gradio workers might write logs simultaneously. We use
    a threading lock to prevent data corruption. In a production system
    you'd use a database; for local use, file locking is sufficient.
"""

import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ── Configuration ───────────────────────────────────────────────────────

LOG_DIR = Path("data/monitoring")
LOG_FILE = LOG_DIR / "usage_log.jsonl"

# Thread lock for safe concurrent writes
_write_lock = threading.Lock()


# ── Cost Estimation ─────────────────────────────────────────────────────
# Prices per 1M tokens (input/output) for common models.
# Free-tier models are $0. Updated as of 2024.
# Format: model_id_pattern → (input_cost_per_1M, output_cost_per_1M)

MODEL_COSTS = {
    # Groq (free tier)
    "llama-3.1-8b-instant": (0.0, 0.0),
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "gemma2-9b-it": (0.0, 0.0),
    "mixtral-8x7b-32768": (0.0, 0.0),
    # OpenRouter free models
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "mistralai/mistral-7b-instruct:free": (0.0, 0.0),
    "google/gemma-2-9b-it:free": (0.0, 0.0),
    "qwen/qwen-2.5-72b-instruct:free": (0.0, 0.0),
    # Google Gemini (free tier)
    "gemini-1.5-flash": (0.0, 0.0),
    "gemini-1.5-pro": (0.0, 0.0),
    "gemini-2.0-flash": (0.0, 0.0),
    # Ollama (local, free)
    "ollama/llama3.1": (0.0, 0.0),
    "ollama/mistral": (0.0, 0.0),
    "ollama/phi3": (0.0, 0.0),
}

# Default cost for unknown models (conservative estimate)
DEFAULT_COST = (0.50, 1.50)  # $0.50/M input, $1.50/M output


# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    """
    A single LLM interaction record.

    Every time we call an LLM (chat, agent, etc.), we create one of these
    and save it to the log. This is our core monitoring data structure.

    Attributes:
        timestamp: When the interaction happened (ISO format).
        model: The model ID used (e.g., "llama-3.1-8b-instant").
        provider: Which provider handled it (e.g., "groq", "openrouter").
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the response.
        total_tokens: Sum of input + output.
        estimated_cost: Estimated cost in USD.
        latency_ms: Response time in milliseconds.
        success: Whether the call succeeded.
        error: Error message if it failed (None otherwise).
        feature: Which feature made the call ("chat", "agent", "rag").
    """
    timestamp: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    feature: str = "chat"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        self.total_tokens = self.input_tokens + self.output_tokens


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an LLM call based on the model and token counts.

    Args:
        model: The model ID.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD. Returns 0.0 for free-tier models.
    """
    input_rate, output_rate = MODEL_COSTS.get(model, DEFAULT_COST)

    cost = (
        (input_tokens / 1_000_000) * input_rate
        + (output_tokens / 1_000_000) * output_rate
    )
    return round(cost, 6)


# ── Logging Functions ───────────────────────────────────────────────────

def log_usage(record: UsageRecord) -> None:
    """
    Append a usage record to the log file.

    Thread-safe: uses a lock to prevent corruption from concurrent writes.

    Args:
        record: The UsageRecord to log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")


def track_llm_call(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool = True,
    error: str | None = None,
    feature: str = "chat",
) -> UsageRecord:
    """
    Create a usage record and log it. Convenience function.

    Args:
        model: Model ID.
        provider: Provider name.
        input_tokens: Input token count.
        output_tokens: Output token count.
        latency_ms: Response time in ms.
        success: Whether the call succeeded.
        error: Error message if failed.
        feature: Feature that made the call.

    Returns:
        The created UsageRecord.
    """
    record = UsageRecord(
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
        success=success,
        error=error,
        feature=feature,
    )

    log_usage(record)
    return record


# ── Timer Context Manager ──────────────────────────────────────────────

class Timer:
    """
    A simple context manager for measuring elapsed time.

    Usage:
        with Timer() as t:
            response = call_llm(...)
        print(f"Took {t.elapsed_ms}ms")

    WHY A CONTEXT MANAGER?
        It ensures we always measure the full duration, even if the call
        raises an exception. The elapsed time is available on the Timer
        object after the `with` block exits.
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False  # Don't suppress exceptions


# ── Analytics Functions ─────────────────────────────────────────────────

def load_all_records() -> list[dict]:
    """
    Load all usage records from the log file.

    Returns:
        List of record dicts. Empty list if no log exists.
    """
    if not LOG_FILE.exists():
        return []

    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines

    return records


def get_summary_stats() -> dict:
    """
    Calculate summary statistics from all logged usage.

    Returns:
        Dict with:
            - total_calls: Total LLM interactions
            - total_tokens: Total tokens used (input + output)
            - total_cost: Total estimated cost
            - avg_latency_ms: Average response time
            - success_rate: Percentage of successful calls
            - by_model: Breakdown by model (calls, tokens, cost)
            - by_provider: Breakdown by provider
            - by_feature: Breakdown by feature (chat, agent, rag)
    """
    records = load_all_records()

    if not records:
        return {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "success_rate": 100.0,
            "by_model": {},
            "by_provider": {},
            "by_feature": {},
        }

    total_calls = len(records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_cost = sum(r.get("estimated_cost", 0) for r in records)
    total_latency = sum(r.get("latency_ms", 0) for r in records)
    success_count = sum(1 for r in records if r.get("success", True))

    # Breakdowns
    by_model: dict[str, dict] = {}
    by_provider: dict[str, dict] = {}
    by_feature: dict[str, dict] = {}

    for r in records:
        # By model
        model = r.get("model", "unknown")
        if model not in by_model:
            by_model[model] = {"calls": 0, "tokens": 0, "cost": 0.0}
        by_model[model]["calls"] += 1
        by_model[model]["tokens"] += r.get("total_tokens", 0)
        by_model[model]["cost"] += r.get("estimated_cost", 0)

        # By provider
        provider = r.get("provider", "unknown")
        if provider not in by_provider:
            by_provider[provider] = {"calls": 0, "tokens": 0, "cost": 0.0}
        by_provider[provider]["calls"] += 1
        by_provider[provider]["tokens"] += r.get("total_tokens", 0)
        by_provider[provider]["cost"] += r.get("estimated_cost", 0)

        # By feature
        feature = r.get("feature", "chat")
        if feature not in by_feature:
            by_feature[feature] = {"calls": 0, "tokens": 0, "cost": 0.0}
        by_feature[feature]["calls"] += 1
        by_feature[feature]["tokens"] += r.get("total_tokens", 0)
        by_feature[feature]["cost"] += r.get("estimated_cost", 0)

    return {
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "avg_latency_ms": round(total_latency / total_calls, 1) if total_calls else 0,
        "success_rate": round((success_count / total_calls) * 100, 1) if total_calls else 100.0,
        "by_model": by_model,
        "by_provider": by_provider,
        "by_feature": by_feature,
    }
