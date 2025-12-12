"""Logging utilities."""
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import get_settings

settings = get_settings()
LOG_DIR = Path(settings.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.datetime.now().isoformat()


def write_detailed_log(entry: Dict[str, Any], username: Optional[str] = None, conversation_id: Optional[str] = None):
    """Write detailed log entry to JSONL file in username subfolder.

    Automatically enriches each entry with experiment/config metadata so that
    later analysis can group logs by run.
    
    If conversation_id is provided, writes to conv_{conversation_id}_detailed.jsonl.
    Otherwise, writes to api_detailed.jsonl (legacy behavior).
    """
    # Attach experiment metadata if not already present
    enriched = dict(entry)
    enriched.setdefault("experiment_tag", settings.EXPERIMENT_TAG)
    enriched.setdefault("model", settings.MODEL)
    enriched.setdefault("api_type", settings.API_TYPE)
    enriched.setdefault("use_extra_context", settings.USE_EXTRA_CONTEXT)
    # Use username subfolder if provided, otherwise use 'system' folder
    user_folder = username if username else "system"
    user_log_dir = LOG_DIR / user_folder
    user_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Use per-conversation file if conversation_id is provided
    if conversation_id:
        log_file = user_log_dir / f"conv_{conversation_id}_detailed.jsonl"
    else:
        log_file = user_log_dir / "api_detailed.jsonl"
    
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def write_conversation_log(conversation_id: str, role: str, content: str, username: Optional[str] = None):
    """Write conversation message to text file (similar to mindora_cli_pro.py)."""
    # Use username subfolder if provided, otherwise use 'system' folder
    user_folder = username if username else "system"
    user_log_dir = LOG_DIR / user_folder
    user_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create conversation log file: conv_{conversation_id}.txt
    log_file = user_log_dir / f"conv_{conversation_id}.txt"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {role.upper()}:\n{content.strip()}\n\n")


def save_conversation_snapshot(conversation, user, username: Optional[str] = None):
    """Save conversation snapshot as JSON (similar to mindora_cli_pro.py)."""
    from app.config import get_settings
    
    settings = get_settings()
    
    # Use username subfolder if provided, otherwise use 'system' folder
    user_folder = username if username else "system"
    user_log_dir = LOG_DIR / user_folder
    user_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Build user_bio
    given_name = " ".join(filter(None, [user.first_name, user.last_name])) or "User"
    user_bio = {
        "userId": user.id,
        "givenName": given_name,
        "sex": user.sex,
        "ageYears": user.age_years,
        "religion": user.religion,
        "languages": user.languages or ["en"],
        "location": user.location or {},
        "timeZone": user.time_zone
    }
    
    # Convert messages to turns format
    turns = []
    for msg in conversation.messages:
        turns.append({
            "role": msg.role.value,
            "content": msg.content,
            "at": msg.created_at.isoformat() if msg.created_at else now_iso()
        })
    
    # Extract summaries from summary_note if available
    summaries = []
    if conversation.summary_note and isinstance(conversation.summary_note, dict):
        bullets = conversation.summary_note.get("bullets", [])
        if bullets:
            summaries = bullets if isinstance(bullets, list) else [bullets]
    
    # Calculate metrics
    metrics_data = conversation.metrics or {}
    total_turns_with_kb = len([r for r in metrics_data.get("kb_retrievals", []) if r.get("cards_returned", 0) > 0])
    total_turns = metrics_data.get("total_turns", len(turns))
    pct_turns_with_kb = (total_turns_with_kb / total_turns * 100) if total_turns > 0 else 0.0
    total_cards_returned = sum(r.get("cards_returned", 0) for r in metrics_data.get("kb_retrievals", []))
    request_times = metrics_data.get("request_times", [])
    avg_request_time = (sum(r.get("time_seconds", 0) for r in request_times) / len(request_times)) if request_times else 0.0
    
    # Build snapshot
    snapshot = {
        "user_bio": user_bio,
        "diagnostic_slots": conversation.diagnostic_slots or {},
        "summary_note": conversation.summary_note or {},
        "turns": turns,
        "summaries": summaries,
        "conv_id": conversation.id,
        "updated_at": (conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat()) if conversation.created_at else now_iso(),
        "experiment_config": {
            "experiment_tag": settings.EXPERIMENT_TAG,
            "model": settings.MODEL,
            "summarizer_model": settings.SUMMARIZER_MODEL,
            "safety_model": settings.SAFETY_MODEL,
            "api_type": settings.API_TYPE,
            "use_extra_context": settings.USE_EXTRA_CONTEXT,
            "kb_retrieval_k": settings.KB_RETRIEVAL_K,
            "num_ctx": settings.NUM_CTX,
            "n_turns_to_keep": settings.N_TURNS_TO_KEEP,
            "llm_temperature": settings.LLM_TEMPERATURE,
            "llm_top_p": settings.LLM_TOP_P,
        },
        "metrics": {
            "pct_turns_with_kb": round(pct_turns_with_kb, 2),
            "total_cards_returned": total_cards_returned,
            "request_times": request_times,
            "avg_request_time_seconds": round(avg_request_time, 3),
            "kb_retrievals": metrics_data.get("kb_retrievals", []),
            "total_turns": total_turns
        }
    }
    
    # Save to JSON file: conv_{conversation_id}.json
    json_file = user_log_dir / f"conv_{conversation.id}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

