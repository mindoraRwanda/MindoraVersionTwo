"""Core endpoints (health, logs, etc.)."""
from fastapi import APIRouter, Query
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from app.config import get_settings

router = APIRouter(tags=["core"])
settings = get_settings()


@router.get("/status")
def health_check():
    """API health endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@router.get("/logs")
def get_logs(
    username: Optional[str] = Query(None, description="Username to get logs for (optional)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of log entries to return")
) -> Dict[str, Any]:
    """Get latest logs. If username is provided, returns logs for that user."""
    
    # Use username subfolder if provided, otherwise use 'system' folder
    user_folder = username if username else "system"
    log_file = Path(settings.LOG_DIR) / user_folder / "api_detailed.jsonl"
    
    if not log_file.exists():
        return {"logs": [], "total": 0, "username": user_folder}
    
    logs = []
    try:
        # Read last N lines
        with log_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            # Get last N lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            for line in recent_lines:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": str(e), "logs": [], "total": 0, "username": user_folder}
    
    return {
        "logs": logs,
        "total": len(logs),
        "username": user_folder
    }

