# # services/crisis_classifier.py
#new
import os
import json
import re
from typing import Literal, TypedDict, Dict, Any
from groq import Groq

from datetime import datetime

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

MODEL = os.getenv("CRISIS_MODEL", "llama-3.1-8b-instant")  # safer default than legacy llama3-8b-8192

SYSTEM_PROMPT = """You are a safety classifier for a mental-health chatbot.
Return STRICT JSON with keys: label, severity, confidence, rationale.
label ∈ {self_harm, suicide_ideation, self_injury, substance_abuse, violence, medical_emergency, other, none}
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
    return {}

def classify_crisis(text: str) -> CrisisResult:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        # Avoid response_format to sidestep 400s on some models;
        # enforce JSON via prompt and parse defensively.
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text[:4000])},
            ],
        )
        raw = completion.choices[0].message.content  # attribute access is correct for Groq SDK
        data = _safe_json_extract(raw)
    except Exception as e:
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
