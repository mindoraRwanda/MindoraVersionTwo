"""Safety checking service using Llama Guard."""
import json
import time
import requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import get_settings
from app.utils.logging import write_detailed_log, now_iso

settings = get_settings()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def classify_safety(text: str, username: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
    """Return 'safe' or 'unsafe:<category>' strings from llama-guard output."""
    start_time = time.time()
    body = {
        "model": settings.SAFETY_MODEL,
        "prompt": f"Classify safety of this text:\n{text}",
        "options": {"temperature": 0}
    }

    # Log input
    write_detailed_log({
        "type": "safety_check_input",
        "timestamp": now_iso(),
        "input": text,
        "model": settings.SAFETY_MODEL
    }, username=username, conversation_id=conversation_id)

    r = requests.post(f"{settings.OLLAMA_API_URL}/api/generate", json=body, timeout=30)
    r.raise_for_status()
    out = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            j = json.loads(line)
            out += j.get("response", "")
        except json.JSONDecodeError:
            pass

    # Very simple parse: look for 'unsafe' or 'safe'
    lower = out.strip().lower()
    result = "unsafe" if "unsafe" in lower else "safe"

    elapsed = time.time() - start_time

    # Log output
    write_detailed_log({
        "type": "safety_check_output",
        "timestamp": now_iso(),
        "input": text,
        "output": result,
        "raw_response": out,
        "time_seconds": round(elapsed, 3)
    }, username=username, conversation_id=conversation_id)

    return result

