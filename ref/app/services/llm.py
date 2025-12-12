"""LLM service for structured chat and summarization."""
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, ValidationError
from app.config import get_settings
from app.services.kb import retrieve_kb_hybrid
from app.utils.logging import write_detailed_log, now_iso
from app.models.user import User

settings = get_settings()

# Assistant schema for structured output
ASSISTANT_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "slotUpdates": {
            "type": "object",
            "properties": {
                "sleepIssues": {"type": "string", "enum": ["yes", "no", "unknown"]},
                "lastSleptHours": {"type": ["integer", "null"], "minimum": 0, "maximum": 24},
                "exerciseWeekly": {"type": ["integer", "null"], "minimum": 0, "maximum": 14},
                "appetiteChange": {"type": "string", "enum": ["increase", "decrease", "none", "unknown"]},
                "concentrationIssues": {"type": "string", "enum": ["yes", "no", "unknown"]},
                "socialSupport": {"type": "string", "enum": ["none", "low", "moderate", "strong", "unknown"]},
                "stressors": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["exams", "assignments", "family_conflict", "bullying", "financial", "grief", "health", "relationship", "other"]
                    }
                }
            },
            "additionalProperties": False
        },
        "riskAssessment": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "immediateActions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["level"]
        },
        "nextSteps": {
            "type": "object",
            "properties": {
                "selfHelp": {"type": "array", "items": {"type": "string"}},
                "reachOut": {"type": "array", "items": {"type": "string"}},
                "referral": {"type": "string"}
            }
        },
        "question_next": {"type": "string"},
        "languageHint": {"type": "string", "enum": ["en"]}
    },
    "required": ["message", "question_next", "riskAssessment"]
}


class AssistantTurnState(BaseModel):
    """Structured assistant response."""
    message: str
    slotUpdates: Dict[str, Any] = Field(default_factory=dict)
    riskAssessment: Dict[str, Any]
    nextSteps: Dict[str, Any] = Field(default_factory=dict)
    question_next: str
    languageHint: Optional[str] = None


def build_system_rules(user: User, diagnostic_slots: Dict[str, Any]) -> str:
    """Build system rules for the LLM."""
    # Combine first_name and last_name for givenName in user_bio
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
    
    return f"""
You are Mindora, a supportive mental-health assistant for Rwandan youths (English only).
• Ask EXACTLY ONE diagnostic question per turn. Keep responses ≤ 2 short sentences.
• No diagnosis; give self-help and next steps. Escalate safety if needed.

OUTPUT CONTRACT (must follow strictly):
1) Return VALID JSON per schema.
2) Always include a "slotUpdates" object EVERY TURN (use {{}} if no change).
3) Use ONLY these enums when updating:
   - sleepIssues: ["yes","no","unknown"]
   - appetiteChange: ["increase","decrease","none","unknown"]
   - concentrationIssues: ["yes","no","unknown"]
   - socialSupport: ["none","low","moderate","strong","unknown"]
   - stressors: array of ["exams","assignments","family_conflict","bullying","financial","grief","health","relationship","other"]
   - lastSleptHours: integer 0..24
   - exerciseWeekly: integer 0..14
4) LAST-TURN WINS: If the latest user message contradicts earlier information, OVERWRITE the slot with the new value and include it in "slotUpdates".
5) IMPORTANT: The "message" field should contain your response/acknowledgment (≤2 sentences). The "question_next" field should contain ONLY the diagnostic question. DO NOT have any questions in the message.

User bio: {json.dumps(user_bio, ensure_ascii=False)}
Diagnostic slots: {json.dumps(diagnostic_slots, ensure_ascii=False)}
"""


def build_messages(
    user: User,
    messages: List[Dict[str, str]],
    diagnostic_slots: Dict[str, Any],
    summaries: Optional[List[str]] = None,
    username: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Build message list for LLM with context.
    
    Returns:
        Tuple of (message_list, context_info) where context_info contains:
        - kb_retrieval: dict with cards_returned, method, time_seconds, query (if KB was retrieved)
        - summarize_time: float (if summarization occurred)
        - summaries: list of summary strings (if summaries exist or were generated)
    """
    if summaries is None:
        summaries = []
    
    context_info = {}
    
    recent = messages[-settings.N_TURNS_TO_KEEP:]
    overflow = messages[:-settings.N_TURNS_TO_KEEP] if len(messages) > settings.N_TURNS_TO_KEEP else []
    
    chunks = [
        {"role": "system", "content": build_system_rules(user, diagnostic_slots)}
    ]
    
    # KB snippets from last user turn (only if USE_EXTRA_CONTEXT is enabled)
    if settings.USE_EXTRA_CONTEXT:
        last_user = next((t for t in reversed(recent) if t["role"] == "user"), None)
        if last_user:
            kb_hits, kb_method, kb_time = retrieve_kb_hybrid(
                last_user["content"],
                k=settings.KB_RETRIEVAL_K,
                username=username,
                conversation_id=conversation_id,
            )
            context_info["kb_retrieval"] = {
                "cards_returned": len(kb_hits),
                "method": kb_method,
                "time_seconds": round(kb_time, 3),
                "query": last_user["content"]
            }
            if kb_hits:
                blurb = "\n\n".join([f"[KB] {c.get('title')}: {c.get('bot_say')}" for c in kb_hits])
                chunks.append({"role": "system", "content": "Helpful KB snippets:\n" + blurb})
            
            chunks.append({
                "role": "system",
                "content": "When extracting slotUpdates, use ONLY these values:\n"
                          "sleepIssues: ['yes','no','unknown']; "
                          "appetiteChange: ['increase','decrease','none','unknown']; "
                          "concentrationIssues: ['yes','no','unknown']; "
                          "socialSupport: ['none','low','moderate','strong','unknown']; "
                          "stressors: array of ['exams','assignments','family_conflict','bullying','financial','grief','health','relationship','other']."
            })
        
        # Filter out invalid summaries (e.g., "I can't provide a summary" messages)
        invalid_phrases = [
            "i can't provide a summary",
            "can't provide a summary",
            "our conversation has just begun",
            "conversation has just begun",
            "(no summary)"
        ]
        valid_summaries = [
            s for s in summaries 
            if s and not any(phrase.lower() in s.lower() for phrase in invalid_phrases)
        ]
        
        # Rolling summary
        # Only summarize if there's sufficient overflow (at least 3 messages) to make summarization meaningful
        MIN_OVERFLOW_FOR_SUMMARY = 3
        if overflow and len(overflow) >= MIN_OVERFLOW_FOR_SUMMARY and not valid_summaries:
            overflow_text = "\n".join([f"{t['role']}: {t['content']}" for t in overflow])
            summarize_start = time.time()
            s = summarize_text(overflow_text, username=username, conversation_id=conversation_id)
            summarize_elapsed = time.time() - summarize_start
            context_info["summarize_time"] = round(summarize_elapsed, 3)
            valid_summaries.append(s)
        if valid_summaries:
            chunks.append({"role": "system", "content": "Working summary: " + valid_summaries[-1]})
            # Include summaries in context_info so they can be saved
            context_info["summaries"] = valid_summaries
    
    return chunks + recent, context_info


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.5, max=6),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def chat_structured(messages: List[Dict[str, str]], username: Optional[str] = None, conversation_id: Optional[str] = None) -> AssistantTurnState:
    """Chat with structured outputs, supporting both Ollama and OpenAI-compatible APIs."""
    start_time = time.time()
    
    # Log input
    write_detailed_log({
        "type": "chat_structured_input",
        "timestamp": now_iso(),
        "messages": messages,
        "model": settings.MODEL,
        "api_type": settings.API_TYPE
    }, username=username, conversation_id=conversation_id)
    
    headers = {}
    if settings.API_KEY:
        headers["Authorization"] = f"Bearer {settings.API_KEY}"
    
    if settings.API_TYPE == "openai":
        # OpenAI-compatible API
        url = f"{settings.API_BASE_URL}/v1/chat/completions"
        body = {
            "model": settings.MODEL,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "assistant_response",
                    "schema": ASSISTANT_SCHEMA,
                    "strict": True
                }
            }
        }
        r = requests.post(url, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        # Ollama API
        body = {
            "model": settings.MODEL,
            "messages": messages,
            "think": False,
            "stream": False,
            "format": ASSISTANT_SCHEMA,
            "options": {
                "num_ctx": settings.NUM_CTX,
                "temperature": settings.LLM_TEMPERATURE,
                "top_p": settings.LLM_TOP_P,
            },
        }
        r = requests.post(f"{settings.API_BASE_URL}/api/chat", json=body, headers=headers, timeout=60)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
    
    elapsed = time.time() - start_time
    
    try:
        payload = json.loads(content)
        result = AssistantTurnState(**payload)
        
        # Log output
        write_detailed_log({
            "type": "chat_structured_output",
            "timestamp": now_iso(),
            "input_messages": messages,
            "output": payload,
            "time_seconds": round(elapsed, 3)
        }, username=username, conversation_id=conversation_id)
        
        return result
    except (json.JSONDecodeError, ValidationError) as e:
        # Log error
        write_detailed_log({
            "type": "chat_structured_error",
            "timestamp": now_iso(),
            "input_messages": messages,
            "error": str(e),
            "raw_response": content,
            "time_seconds": round(elapsed, 3)
        }, username=username, conversation_id=conversation_id)
        raise RuntimeError(f"Invalid structured output: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(),
    retry=retry_if_exception_type(requests.RequestException)
)
def summarize_text(text: str, username: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
    """Summarize text, supporting both Ollama and OpenAI-compatible APIs."""
    start_time = time.time()
    prompt = ("You must summarize the following chat history into <=6 bullet facts (present tense). "
              "Focus on symptoms, stressors, protective factors, and any risks. "
              "Always provide a factual summary based on the conversation content.\n\n" + text)
    
    # Log input
    write_detailed_log({
        "type": "summarize_input",
        "timestamp": now_iso(),
        "input_text": text,
        "model": settings.SUMMARIZER_MODEL,
        "api_type": settings.API_TYPE
    }, username=username, conversation_id=conversation_id)
    
    headers = {}
    if settings.API_KEY:
        headers["Authorization"] = f"Bearer {settings.API_KEY}"
    
    if settings.API_TYPE == "openai":
        # OpenAI-compatible API
        url = f"{settings.API_BASE_URL}/v1/completions"
        body = {
            "model": settings.SUMMARIZER_MODEL,
            "prompt": prompt,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": 500
        }
        r = requests.post(url, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        response_data = r.json()
        out = response_data.get("choices", [{}])[0].get("text", "")
    else:
        # Ollama API
        r = requests.post(
            f"{settings.API_BASE_URL}/api/generate",
            json={
                "model": settings.SUMMARIZER_MODEL,
                "prompt": prompt,
                "options": {
                    "num_ctx": settings.NUM_CTX,
                    "temperature": settings.LLM_TEMPERATURE,
                    "top_p": settings.LLM_TOP_P,
                },
            },
            headers=headers,
            timeout=60,
            stream=False
        )
        r.raise_for_status()
        out = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                out += json.loads(line).get("response", "")
            except Exception:
                pass
    
    result = out.strip() or "(no summary)"
    
    # Validate that the result is not an invalid summary
    invalid_phrases = [
        "i can't provide a summary",
        "can't provide a summary",
        "our conversation has just begun",
        "conversation has just begun"
    ]
    if any(phrase.lower() in result.lower() for phrase in invalid_phrases):
        # If the model refused to summarize, return a fallback
        result = "(no summary)"
    
    elapsed = time.time() - start_time
    
    # Log output
    write_detailed_log({
        "type": "summarize_output",
        "timestamp": now_iso(),
        "input_text": text,
        "output": result,
        "time_seconds": round(elapsed, 3)
    }, username=username, conversation_id=conversation_id)
    
    return result

