"""Cross-process token usage ledger for KernelAgent LLM calls.

Workers are spawned with the ``spawn`` multiprocessing context, so an
in-memory counter in the parent process cannot observe calls made in worker
subprocesses. This module uses a single append-only JSONL file (path selected
by the ``KERNELAGENT_TOKEN_LEDGER`` environment variable) that every worker
process inherits (spawn snapshots ``os.environ`` at ``Process.start()``) and
appends one line per LLM call. Appends are atomic at the OS level, so no lock
is required.
"""

import json
import os
from pathlib import Path


def ledger_path() -> Path | None:
    """Return the configured ledger file path, or None when tracing is disabled."""
    raw = os.environ.get("KERNELAGENT_TOKEN_LEDGER")
    if not raw:
        return None
    return Path(raw)


def reset() -> None:
    """Truncate the ledger file (called by the bridge before a problem run)."""
    path = ledger_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def record(usage: dict | None) -> None:
    """Append one LLM call's usage as a JSON line (called from providers)."""
    if not usage:
        return
    path = ledger_path()
    if path is None:
        return
    line = json.dumps(
        {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
        ensure_ascii=False,
    )
    # 'a' mode appends atomically for small line writes on POSIX and Win32.
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sum_tokens() -> dict:
    """Sum every recorded call's token usage into a single aggregate."""
    path = ledger_path()
    if path is None or not path.is_file():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc["prompt_tokens"] += int(row.get("prompt_tokens", 0) or 0)
            acc["completion_tokens"] += int(row.get("completion_tokens", 0) or 0)
            acc["total_tokens"] += int(row.get("total_tokens", 0) or 0)
    except OSError:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return acc
