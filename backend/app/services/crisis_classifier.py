# # services/crisis_classifier.py
#new
import os
import json
import re
import logging
from typing import Literal, TypedDict, Dict, Any, Optional
from .llm_providers import create_llm_provider

from datetime import datetime

logger = logging.getLogger(__name__)

CrisisLabel = Literal[
    "self_harm", "suicide_ideation", "self_injury",
    "substance_abuse", "violence", "medical_emergency", "other", "none"
]
Severity = Literal["low", "moderate", "high", "imminent"]

class CrisisResult(TypedDict, total=False):
    label: CrisisLabel
    severity: Severity
    confidence: float
    rationale: str

SYSTEM_PROMPT = """You are a safety classifier for a mental-health chatbot.
Return STRICT JSON with keys: label, severity, confidence, rationale.
label ∈ {"self_harm", "suicide_ideation", "self_injury",
    "substance_abuse", "violence", "medical_emergency", "other", "none"}
severity ∈ {low, moderate, high, imminent} (if label is 'none' use 'low')
confidence ∈ [0,1]
Respond ONLY with JSON.
"""

USER_TEMPLATE = """Text: {text}
Consider Rwanda/East Africa context and slang. Detect intent, not just keywords.
Return ONLY JSON.
"""

_ALLOWED_LABELS = {
    "self_harm", "suicide_ideation", "self_injury",
    "substance_abuse", "violence", "medical_emergency", "other", "none"
}
_ALLOWED_SEVERITIES = {"low", "moderate", "high", "imminent"}

def _safe_json_extract(raw: str) -> Dict[str, Any]:
    # First try direct JSON
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Then try to extract the first {...} block
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Try to clean markdown code blocks
    try:
        if "```json" in raw:
            clean = raw.split("```json")[1].split("```")[0].strip()
            return json.loads(clean)
        elif "```" in raw:
            clean = raw.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
    except Exception:
        pass
        
    return {}

async def classify_crisis(text: str, provider_name: Optional[str] = None, model_name: Optional[str] = None) -> CrisisResult:
    try:
        # Import settings to get configuration
        from ..settings.settings import settings
        
        # Use settings configuration if not explicitly provided
        if provider_name is None:
            provider_name = settings.model.llm_provider if settings.model else None
        if model_name is None:
            model_name = settings.model.model_name if settings.model else "gemma3:1b"
            
        # Create provider using settings configuration
        llm_provider = create_llm_provider(provider=provider_name, model=model_name)
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
        except (ImportError, BaseException):
            class SystemMessage:
                def __init__(self, content): self.content = content
            class HumanMessage:
                def __init__(self, content): self.content = content
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(text=text[:4000])),
        ]
        
        raw = await llm_provider.generate_response(messages)
        data = _safe_json_extract(raw)
    except Exception as e:
        logger.error(f"Crisis classification failed: {e}")
        return {"label": "other", "severity": "low", "confidence": 0.0, "rationale": f"llm_error:{e}"}

    # Normalize and validate
    label = str(data.get("label", "other")).strip().lower().replace(" ", "_")
    severity = str(data.get("severity", "low")).strip().lower()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    rationale = str(data.get("rationale", ""))

    if label not in _ALLOWED_LABELS:
        label = "other"
    if severity not in _ALLOWED_SEVERITIES:
        severity = "low"

    return {"label": label, "severity": severity, "confidence": confidence, "rationale": rationale}
