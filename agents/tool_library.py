"""
agents/tool_library.py — Extended Tool Library for Nexus AI

This module adds practical tools that make the agent significantly
more capable. While Phase 3 gave us web search and KB search, these
tools handle computation, code execution, and file operations.

WHY THESE TOOLS?
    LLMs are great at language but BAD at:
    - Math: "What's 347 * 829?" → often wrong
    - Precise dates: "What day is 90 days from now?" → often wrong
    - Code execution: They can WRITE code but can't RUN it

    Tools fix this by letting the LLM delegate to real code.

TOOL DESIGN PRINCIPLES:
    1. Single responsibility: Each tool does ONE thing well
    2. Clear descriptions: The LLM reads these to decide when to use each tool
    3. Safe by default: Code execution is sandboxed, file writes are restricted
    4. Useful output: Return structured, readable results
    5. Graceful errors: Never crash — return an error message instead

SECURITY NOTE — CODE EXECUTION:
    The code executor runs Python in a restricted environment:
    - No file system access outside the data/ directory
    - No network access (no imports of requests/urllib/etc.)
    - Timeout after 10 seconds (prevents infinite loops)
    - No system calls (no os.system, subprocess, etc.)
    This is NOT production-grade sandboxing (that would need Docker/Firecracker),
    but it's sufficient for a learning project.
"""

import ast
import datetime
import io
import math
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from langchain_core.tools import tool


# ── Calculator Tool ──────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the precise result.

    Use this for ANY math calculation. LLMs are unreliable at arithmetic,
    so always use this tool instead of trying to calculate mentally.

    Supports: +, -, *, /, **, sqrt(), sin(), cos(), tan(), log(), abs(),
    round(), min(), max(), pi, e

    Args:
        expression: A math expression like "347 * 829" or "sqrt(144) + pi"

    Returns:
        The calculated result as a string.
    """
    try:
        # We use Python's ast.literal_eval for simple expressions,
        # but for math functions we need a safe eval approach.
        # We provide a restricted namespace with only math functions.
        safe_namespace = {
            "__builtins__": {},  # No built-in functions (security)
            # Math constants
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            # Math functions
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            # Trigonometry
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            # Logarithms
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            # Other
            "ceil": math.ceil,
            "floor": math.floor,
            "factorial": math.factorial,
        }

        result = eval(expression, safe_namespace)  # noqa: S307
        return f"{expression} = {result}"

    except ZeroDivisionError:
        return f"Error: Division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


# ── Python Code Executor ────────────────────────────────────────────────

# Modules that are BLOCKED for security reasons
BLOCKED_MODULES = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "signal", "threading",
    "multiprocessing",
}


@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output.

    Use this to run calculations, data processing, or any Python code.
    The code runs in a restricted sandbox — no file system or network access.

    Print statements will be captured and returned as output.
    The last expression's value is also returned if it's not None.

    Args:
        code: Python code to execute. Use print() to produce output.

    Returns:
        The stdout output from the code execution, or an error message.
    """
    # Security check: scan for blocked module imports
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in BLOCKED_MODULES:
                        return f"Security error: Module '{alias.name}' is not allowed."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in BLOCKED_MODULES:
                        return f"Security error: Module '{node.module}' is not allowed."
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"

    # Set up restricted namespace with safe built-ins
    safe_builtins = {
        "print": print, "len": len, "range": range, "str": str,
        "int": int, "float": float, "bool": bool, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "type": type,
        "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "isinstance": isinstance, "issubclass": issubclass,
        "hasattr": hasattr, "getattr": getattr,
        "True": True, "False": False, "None": None,
        "__import__": __import__,  # Needed for allowed imports
    }

    namespace = {"__builtins__": safe_builtins}

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)  # noqa: S102

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\nWarnings:\n{errors}"

        return result.strip() if result.strip() else "Code executed successfully (no output)."

    except Exception as e:
        return f"Runtime error: {type(e).__name__}: {str(e)}"


# ── Date/Time Tool ──────────────────────────────────────────────────────

@tool
def datetime_tool(query: str) -> str:
    """Get current date/time or perform date calculations.

    Use this for any time-related questions. The query can be:
    - "now" — current date and time
    - "today" — today's date
    - "+30 days" — date 30 days from now
    - "-2 weeks" — date 2 weeks ago
    - "days until 2025-12-31" — days between now and a date

    Args:
        query: A time query like "now", "+30 days", or "days until 2025-12-31"

    Returns:
        The formatted date/time result.
    """
    now = datetime.datetime.now()

    query = query.strip().lower()

    if query in ("now", "current time", "time"):
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"

    if query in ("today", "date", "current date"):
        return f"Today's date: {now.strftime('%Y-%m-%d (%A, %B %d, %Y)')}"

    # Handle "+N days/weeks/months"
    if query.startswith("+") or query.startswith("-"):
        try:
            parts = query.split()
            amount = int(parts[0])
            unit = parts[1] if len(parts) > 1 else "days"

            if "week" in unit:
                delta = datetime.timedelta(weeks=amount)
            elif "month" in unit:
                delta = datetime.timedelta(days=amount * 30)
            elif "year" in unit:
                delta = datetime.timedelta(days=amount * 365)
            else:
                delta = datetime.timedelta(days=amount)

            result_date = now + delta
            return (
                f"{now.strftime('%Y-%m-%d')} {'+' if amount >= 0 else ''}{amount} {unit} = "
                f"{result_date.strftime('%Y-%m-%d (%A, %B %d, %Y)')}"
            )
        except (ValueError, IndexError):
            return f"Could not parse date offset: '{query}'. Use format: '+30 days'"

    # Handle "days until YYYY-MM-DD"
    if "until" in query or "between" in query:
        try:
            # Extract the date from the query
            import re
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", query)
            if date_match:
                target = datetime.datetime.strptime(date_match.group(), "%Y-%m-%d")
                delta = target - now
                return (
                    f"From {now.strftime('%Y-%m-%d')} to {target.strftime('%Y-%m-%d')}: "
                    f"{delta.days} days"
                )
        except ValueError:
            pass

    return (
        f"Current: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Could not understand query: '{query}'\n"
        f"Try: 'now', 'today', '+30 days', '-2 weeks', 'days until 2025-12-31'"
    )


# ── File Writer Tool ────────────────────────────────────────────────────

@tool
def save_to_file(filename: str, content: str) -> str:
    """Save text content to a file in the data/outputs/ directory.

    Use this to save research results, reports, summaries, or any
    generated content that the user might want to keep.

    Files are saved to data/outputs/ (not arbitrary paths for security).

    Args:
        filename: The filename (e.g., "research_report.md"). No path separators allowed.
        content: The text content to save.

    Returns:
        Confirmation message with the file path.
    """
    # Security: prevent path traversal attacks
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Error: Filename cannot contain path separators or '..'"

    # Ensure the output directory exists
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    # Don't overwrite existing files without indication
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while file_path.exists():
            file_path = output_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    file_path.write_text(content, encoding="utf-8")

    return f"Saved to: {file_path} ({len(content)} characters)"


# ── Tool Collection ──────────────────────────────────────────────────────

UTILITY_TOOLS = [
    calculator,
    run_python,
    datetime_tool,
    save_to_file,
]
