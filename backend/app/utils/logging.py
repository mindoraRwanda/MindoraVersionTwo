"""
Application logging utilities modeled after the reference app.

Provides:
- ISO timestamp helper
- Structured JSONL logging to per-user/per-conversation files
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..settings import settings


def _get_log_dir() -> Path:
    """
    Resolve the log directory from settings if available, otherwise default to 'logs'.

    This mirrors the reference app's behavior but adapts to the new settings system.
    """
    # Settings may or may not define a dedicated log_dir; fall back safely.
    raw_dir = getattr(settings, "log_dir", None) or "logs"
    path = Path(raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


LOG_DIR = _get_log_dir()


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.datetime.now().isoformat()


def write_detailed_log(
    entry: Dict[str, Any],
    username: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> None:
    """
    Write a structured log entry as JSONL to disk.

    The log path layout mirrors the reference app:
    - logs/<username or 'system'>/api_detailed.jsonl
    - logs/<username or 'system'>/conv_<conversation_id>_detailed.jsonl

    Entries are automatically enriched with key configuration metadata so that
    downstream analysis can group runs by environment and model configuration.
    """
    enriched: Dict[str, Any] = dict(entry)

    # Attach high-level configuration context if not already present
    enriched.setdefault("environment", getattr(settings, "environment", "development"))

    # Model configuration (if available)
    model_cfg = getattr(settings, "model", None)
    if model_cfg is not None:
        enriched.setdefault("model_name", getattr(model_cfg, "model_name", None))
        enriched.setdefault("temperature", getattr(model_cfg, "temperature", None))

    # Performance configuration (if available)
    perf_cfg = getattr(settings, "performance", None)
    if perf_cfg is not None:
        enriched.setdefault("rag_top_k", getattr(perf_cfg, "rag_top_k", None))
        enriched.setdefault("max_input_length", getattr(perf_cfg, "max_input_length", None))

    # Safety configuration (if available)
    safety_cfg = getattr(settings, "safety", None)
    if safety_cfg is not None:
        crisis_kw = getattr(safety_cfg, "crisis_keywords", []) or []
        enriched.setdefault("crisis_keywords_count", len(crisis_kw))

    # Choose user folder (per-user logs like the reference app)
    user_folder = username if username else "system"
    user_log_dir = LOG_DIR / user_folder
    user_log_dir.mkdir(parents=True, exist_ok=True)

    # Choose file based on conversation id
    if conversation_id:
        log_file = user_log_dir / f"conv_{conversation_id}_detailed.jsonl"
    else:
        log_file = user_log_dir / "api_detailed.jsonl"

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")


