"""
llm_checker.py — LLM mentor feedback using LM Studio (OpenAI-compatible API).

Setup:
  1. Open LM Studio -> Local Server -> Start Server (default: http://localhost:1234)
  2. Load any model (recommended: mistral-7b, deepseek-coder, or any instruct model)
  3. pip install openai

Usage in notebooks:
    from llm_checker import check, hint, evaluate, progress, show_exercise

    # Auto-check assertions:
    check([
        (isinstance(result, float), "Returns a float"),
        (result > 0,                "Value is positive"),
    ], exercise_id="01a-2")

    # Get a hint at level 1-3:
    hint(EXERCISE_DESC, level=1)

    # Full evaluate (check + mentor feedback):
    evaluate(my_function, EXERCISE_DESC, tests_fn=lambda fn: check([...]))

    # Show progress across exercises:
    progress()
"""

import inspect
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL = "local-model"          # LM Studio accepts any string when a model is loaded
FALLBACK_WITHOUT_LLM = True    # If True, check() still works when LM Studio is offline
PROGRESS_FILE = Path(".llm_checker_progress.json")

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_client = None
_progress: dict = {}

if PROGRESS_FILE.exists():
    try:
        _progress = json.loads(PROGRESS_FILE.read_text())
    except Exception:
        _progress = {}


def _get_client():
    """Lazy-init OpenAI client; returns None if LM Studio is unreachable."""
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
        c = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=10)
        c.models.list()   # quick connectivity check
        _client = c
        return _client
    except Exception:
        if not FALLBACK_WITHOUT_LLM:
            raise
        return None


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SYSTEM_MENTOR = """\
You are a Socratic Python mentor reviewing student code from a practical LLM engineering course.
Rules (follow strictly):
1. NEVER reveal the complete solution.
2. Ask 1-2 guiding questions to help the student discover issues themselves.
3. Explicitly praise good design decisions (naming, structure, pythonic style).
4. If the code is correct, suggest polish or edge cases to consider.
5. Reply in max 6 sentences. Be warm, encouraging, and direct.
6. Use code snippets only when strictly necessary — prefer prose.
"""

_SYSTEM_HINT = """\
You are a helpful Python tutor giving progressive hints for a coding exercise.
Never reveal the full solution. Be concise (max 4 sentences per hint).
"""


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------
def _llm(prompt: str, system: str = _SYSTEM_MENTOR, max_tokens: int = 450) -> str:
    client = _get_client()
    if client is None:
        return (
            "⚠️  LM Studio is not running (or unreachable at localhost:1234).\n"
            "Start LM Studio, load a model, and re-run this cell to get mentor feedback.\n"
            "Auto-checks above work independently of LM Studio."
        )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️  LLM call failed: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(
    assertions: list,
    exercise_id: str = "",
    *,
    silent: bool = False,
) -> bool:
    """
    Run a list of (condition, description) assertions and print results.

    Parameters
    ----------
    assertions : list of (bool, str)
        Each tuple is (condition_that_should_be_true, human_readable_description).
    exercise_id : str
        Short ID like "01a-2". Used for progress tracking and display.
    silent : bool
        If True, suppress all output (useful when called from evaluate()).

    Returns
    -------
    bool
        True if every assertion passed.

    Example
    -------
    check([
        (isinstance(cost, float),  "estimate_cost returns a float"),
        (cost > 0,                 "Cost is positive"),
        (lm_cost == 0.0,           "lm-studio cost is $0.00"),
    ], exercise_id="01a-2")
    """
    all_pass = True
    lines = []
    for condition, description in assertions:
        icon = "✅" if condition else "❌"
        lines.append(f"  {icon}  {description}")
        if not condition:
            all_pass = False

    if not silent:
        status = "ALL PASSED ✅" if all_pass else "SOME FAILED ❌"
        border = "─" * 54
        label = f"  Exercise {exercise_id}" if exercise_id else "  Auto-check"
        print(f"\n{border}")
        print(f"{label}  →  {status}")
        print(border)
        for line in lines:
            print(line)
        print(border + "\n")

    # Persist progress
    if exercise_id:
        _progress[exercise_id] = {
            "passed": all_pass,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            PROGRESS_FILE.write_text(json.dumps(_progress, indent=2))
        except Exception:
            pass

    return all_pass


def hint(
    exercise_description: str,
    user_code: str = "",
    level: int = 1,
) -> None:
    """
    Print a progressive hint at the given level (1=vague, 2=method names, 3=pseudocode).

    Example
    -------
    hint(EXERCISE_DESC, level=1)
    hint(EXERCISE_DESC, user_code=my_code, level=2)
    """
    detail = {
        1: "Give only a general conceptual direction. No code, no method names.",
        2: "Name the right Python method or data structure to use, but don't show how.",
        3: "Provide short pseudocode (no actual runnable Python) outlining the algorithm.",
    }
    level = max(1, min(3, level))
    prompt = (
        f"EXERCISE:\n{exercise_description}\n\n"
        + (f"STUDENT CODE SO FAR:\n```python\n{user_code}\n```\n\n" if user_code.strip() else "")
        + f"HINT LEVEL {level}/3: {detail[level]}"
    )
    response = _llm(prompt, system=_SYSTEM_HINT, max_tokens=280)
    border = "─" * 54
    print(f"\n{border}")
    print(f"  💡  Hint  (level {level}/3)  — call hint(..., level={min(level+1,3)}) for more detail")
    print(border)
    print(response)
    print(border + "\n")


def get_feedback(
    user_func: Callable,
    exercise_description: str,
    tests_passed: bool,
) -> str:
    """
    Get LLM mentor feedback on a function. Returns the feedback string.
    Usually called indirectly through evaluate().
    """
    try:
        code = inspect.getsource(user_func)
    except Exception:
        code = "<source unavailable>"

    status = "ALL TESTS PASSED ✅" if tests_passed else "SOME TESTS FAILED ❌"
    prompt = (
        f"EXERCISE:\n{exercise_description}\n\n"
        f"TEST STATUS: {status}\n\n"
        f"STUDENT CODE:\n```python\n{code}\n```\n\n"
        "Review: logical correctness, edge cases, readability, pythonic style. "
        "Follow your mentoring rules exactly."
    )
    return _llm(prompt)


def evaluate(
    user_func: Callable,
    exercise_description: str,
    tests_fn: Optional[Callable] = None,
    exercise_id: str = "",
) -> tuple:
    """
    Full evaluation: run tests, then get LLM mentor feedback.

    Parameters
    ----------
    user_func    : the student's function to review
    exercise_description : plain-text description of the exercise
    tests_fn     : callable(user_func) -> bool  (should call check() internally)
    exercise_id  : short ID for progress tracking

    Returns
    -------
    (passed: bool, feedback: str)

    Example
    -------
    def run_tests(fn):
        result = fn("hello world", "gpt-4o")
        return check([
            (isinstance(result, float), "Returns float"),
            (result > 0,               "Positive cost"),
        ], exercise_id="01a-2")

    evaluate(estimate_cost, EXERCISE_DESC, tests_fn=run_tests, exercise_id="01a-2")
    """
    ok = False
    if tests_fn is not None:
        print("🔍 Running auto-checks...\n")
        try:
            ok = tests_fn(user_func)
        except Exception as exc:
            print(f"❌ Test runner raised an exception:\n   {exc}")
            traceback.print_exc()
    else:
        print("ℹ️  No tests_fn provided — skipping auto-checks.\n")

    print("🤖 Fetching mentor feedback from LM Studio...\n")
    t0 = time.perf_counter()
    fb = get_feedback(user_func, exercise_description, ok)
    elapsed = time.perf_counter() - t0

    border = "═" * 54
    print(f"\n{border}")
    print("  🎓  MENTOR FEEDBACK")
    print(border)
    print(fb)
    print(f"\n  (generated in {elapsed:.1f}s via LM Studio)")
    print(border + "\n")

    return ok, fb


def progress() -> None:
    """
    Print a summary table of all recorded exercise attempts.

    Example
    -------
    progress()
    """
    if not _progress:
        print("No exercise attempts recorded yet.")
        return

    border = "─" * 54
    print(f"\n{border}")
    print("  📊  Exercise Progress")
    print(border)
    total = len(_progress)
    n_passed = sum(1 for v in _progress.values() if v.get("passed"))
    for ex_id, info in sorted(_progress.items()):
        icon = "✅" if info.get("passed") else "❌"
        ts = info.get("ts", "")
        print(f"  {icon}  {ex_id:<22}  {ts}")
    print(border)
    pct = (100 * n_passed // total) if total else 0
    print(f"  Total: {n_passed}/{total} passed  ({pct}%)")
    print(border + "\n")


def reset_progress() -> None:
    """Clear all saved progress (e.g. to restart a module)."""
    global _progress
    _progress = {}
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    print("✅ Progress reset.")


# ---------------------------------------------------------------------------
# Notebook display helper
# ---------------------------------------------------------------------------

def show_exercise(
    exercise_id: str,
    title: str,
    description: str,
    hints: list = None,
    checks: list = None,
    exercise_type: str = "EXERCISE",
) -> None:
    """
    Render a formatted exercise card inside a Jupyter notebook cell.

    Parameters
    ----------
    exercise_id   : e.g. "01a-2"
    title         : short title
    description   : full exercise text
    hints         : list of hint strings
    checks        : list of auto-check descriptions
    exercise_type : "EXAMPLE", "EXERCISE", or "CHALLENGE"

    Example
    -------
    show_exercise(
        "01a-2", "Cost estimator for 3 models",
        "Write estimate_cost(text, model) -> float ...",
        hints=["Use tiktoken to count tokens"],
        checks=["Returns float for all 4 models", "lm-studio → $0.00"],
    )
    """
    badge = {"EXAMPLE": "🔵 EXAMPLE", "EXERCISE": "🟡 EXERCISE", "CHALLENGE": "🔴 CHALLENGE"}.get(
        exercise_type, exercise_type
    )
    lines = [
        f"## {badge} &nbsp; Ex {exercise_id} — {title}",
        "",
        description,
        "",
    ]
    if hints:
        lines += ["### 💡 Hints", ""]
        for h in hints:
            lines.append(f"- `{h}`" if h.startswith("▸") else f"- {h}")
        lines.append("")
    if checks:
        lines += ["### ✔️ Auto-check verifies", ""]
        for c in checks:
            lines.append(f"- {c}")
        lines.append("")

    md = "\n".join(lines)
    try:
        from IPython.display import display, Markdown
        display(Markdown(md))
    except ImportError:
        print(md)
