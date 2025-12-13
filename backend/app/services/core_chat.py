"""
Core structured chat engine that mirrors the reference app's behavior while
plugging into the existing LangGraph pipeline and LLM provider stack.

Responsibilities:
- Build a compact, rules-focused system prompt
- Optionally enrich with RAG / knowledge snippets
- Request strictly structured JSON output from the LLM
- Maintain diagnostic slots (sleep, stressors, support, etc.)
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError
import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.logging import write_detailed_log, now_iso
from .diagnostic_slots import get_default_slots, apply_slot_updates

logger = logging.getLogger(__name__)


class AssistantTurnState(BaseModel):
    """
    Structured assistant response schema.

    This closely follows the reference app to keep downstream analytics simple.
    """

    message: str = Field(..., description="Brief empathetic response (1-2 sentences). MUST NOT contain any questions. Questions go in 'question_next' field only.")
    slotUpdates: Dict[str, Any] = Field(default_factory=dict, description="Updates to diagnostic slots based on this turn")
    riskAssessment: Dict[str, Any] = Field(default_factory=dict, description="Risk assessment data")
    nextSteps: Dict[str, Any] = Field(default_factory=dict, description="Recommended next steps")
    question_next: str = Field(..., description="The diagnostic question to ask next. MUST be a complete question sentence. MUST NOT contain responses or statements.")
    languageHint: Optional[str] = Field(None, description="Language preference hint")


def build_system_rules(
    user_context: Dict[str, Any],
    diagnostic_slots: Dict[str, Any],
    strategy: Optional[str],
    emotion: Optional[str],
) -> str:
    """
    System prompt that encodes conversational and output contract rules.

    It is intentionally compact and focused on:
    - single diagnostic question per turn
    - short responses
    - structured JSON output contract
    """
    return (
        "You are Mindora, a supportive mental-health assistant for Rwandan youths (English only).\n"
        "• Ask EXACTLY ONE diagnostic question per turn. Keep responses ≤ 2 short sentences.\n"
        "• No diagnosis; give self-help and next steps. Escalate safety if needed.\n\n"
        "OUTPUT CONTRACT (CRITICAL - MUST FOLLOW EXACTLY):\n"
        "1) Return VALID JSON following the 'assistant_response' schema.\n"
        "2) Always include a 'slotUpdates' object EVERY TURN (use {} if no change).\n"
        "3) Use ONLY the documented enums for slots (sleepIssues, appetiteChange, socialSupport, stressors, etc.).\n"
        "4) LAST-TURN WINS: If the latest user message contradicts earlier information, OVERWRITE the slot.\n"
        "5) CRITICAL SEPARATION:\n"
        "   - 'message' field: Contains ONLY your brief empathetic response (1-2 sentences). NO QUESTIONS ALLOWED.\n"
        "   - 'question_next' field: Contains ONLY the diagnostic question (as a complete question sentence).\n"
        "   - NEVER put questions in the 'message' field. NEVER put responses in the 'question_next' field.\n"
        "   - Example CORRECT format:\n"
        "     message: \"I understand that stress can affect your appetite. That's completely normal.\"\n"
        "     question_next: \"Have you noticed any changes in your sleep patterns recently?\"\n"
        "   - Example WRONG format (DO NOT DO THIS):\n"
        "     message: \"I understand. Have you noticed any sleep changes?\"  ❌ (contains question)\n"
        "     question_next: \"I see. That's normal.\"  ❌ (not a question)\n\n"
        f"Current diagnostic_slots: {diagnostic_slots}\n"
        f"Current emotion (from upstream pipeline): {emotion or 'neutral'}\n"
        f"Strategy hint from upstream routing (optional): {strategy or 'GIVE_EMPATHY'}\n"
        f"User context (may be partial): {user_context}\n"
    )


async def run_core_chat_turn(
    *,
    llm_provider,
    query: str,
    conversation_history: List[Dict[str, Any]],
    diagnostic_slots: Optional[Dict[str, Any]] = None,
    user_context: Optional[Dict[str, Any]] = None,
    emotion: Optional[str] = None,
    strategy: Optional[str] = None,
    rag_text: Optional[str] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[AssistantTurnState, Dict[str, Any]]:
    """
    Run a single structured chat turn using the shared LLM provider.

    Returns:
        (assistant_state, updated_slots)
    """
    slots = diagnostic_slots or get_default_slots()
    user_ctx = user_context or {}

    system_prompt = build_system_rules(
        user_context=user_ctx,
        diagnostic_slots=slots,
        strategy=strategy,
        emotion=emotion,
    )

    messages: List[Any] = [SystemMessage(content=system_prompt)]

    # Optional knowledge snippet from RAG
    if rag_text:
        messages.append(
            SystemMessage(
                content="Helpful knowledge context for this turn (summarized):\n" + rag_text[:800]
            )
        )

    # Add recent conversation snippets as lightweight context
    for turn in conversation_history[-10:]:
        role = turn.get("role", "user")
        content = turn.get("text") or turn.get("content") or ""
        content = str(content).strip()
        if not content:
            continue
        if role == "assistant":
            messages.append(HumanMessage(content=f"(Earlier bot reply for context only, do not repeat): {content}"))
        else:
            messages.append(HumanMessage(content=content))

    # Current user query (primary input for this turn)
    messages.append(HumanMessage(content=query))

    # Log input
    write_detailed_log(
        {
            "type": "core_chat_input",
            "timestamp": now_iso(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_preview": query[:200],
            "history_len": len(conversation_history),
            "slots_before": slots,
            "strategy": strategy,
            "emotion": emotion,
        },
        username=str(user_id) if user_id is not None else None,
        conversation_id=str(conversation_id) if conversation_id is not None else None,
    )

    # Request structured output with retries (like reference app)
    # Retry up to 3 times on validation errors
    max_retries = 3
    retry_delay = 0.5
    last_error = None
    result = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Core chat attempt {attempt + 1}/{max_retries}")
            result: AssistantTurnState = await llm_provider.agenerate(
                messages, structured_output=AssistantTurnState
            )
            # Success - validate we have content
            if not result.message or not result.message.strip():
                logger.warning(f"⚠️  Attempt {attempt + 1}: LLM returned empty message, retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    raise ValueError("LLM returned empty message after all retries")
            
            if not result.question_next or not result.question_next.strip():
                logger.warning(f"⚠️  Attempt {attempt + 1}: LLM returned empty question_next, retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    raise ValueError("LLM returned empty question_next after all retries")
            
            # Success - log if we retried
            if attempt > 0:
                write_detailed_log(
                    {
                        "type": "core_chat_retry_success",
                        "timestamp": now_iso(),
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "attempt": attempt + 1,
                    },
                    username=str(user_id) if user_id is not None else None,
                    conversation_id=str(conversation_id) if conversation_id is not None else None,
                )
            break
            
        except ValidationError as ve:
            last_error = ve
            logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries}: ValidationError - {ve}")
            write_detailed_log(
                {
                    "type": "core_chat_retry_attempt",
                    "timestamp": now_iso(),
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(ve),
                    "error_type": "ValidationError",
                },
                username=str(user_id) if user_id is not None else None,
                conversation_id=str(conversation_id) if conversation_id is not None else None,
            )
            
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                logger.error(f"❌ All {max_retries} retries exhausted for structured output validation")
                write_detailed_log(
                    {
                        "type": "core_chat_retry_exhausted",
                        "timestamp": now_iso(),
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "total_attempts": max_retries,
                        "error": str(ve),
                    },
                    username=str(user_id) if user_id is not None else None,
                    conversation_id=str(conversation_id) if conversation_id is not None else None,
                )
                raise RuntimeError(f"Structured output validation failed after {max_retries} attempts: {ve}") from ve
                
        except Exception as e:
            last_error = e
            logger.error(f"❌ Attempt {attempt + 1}: Unexpected error - {type(e).__name__}: {e}")
            # Only retry on recoverable errors
            if attempt < max_retries - 1 and isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
                write_detailed_log(
                    {
                        "type": "core_chat_retry_attempt",
                        "timestamp": now_iso(),
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "attempt": attempt + 1,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    username=str(user_id) if user_id is not None else None,
                    conversation_id=str(conversation_id) if conversation_id is not None else None,
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                # Don't retry on other errors
                raise
    
    if result is None:
        raise RuntimeError(f"Failed to get valid structured output after {max_retries} attempts: {last_error}") from last_error

    # Post-process to enforce contract: remove questions from message field
    import re
    message_text = result.message.strip()
    question_next_text = result.question_next.strip()
    
    # If message contains questions, extract and move them to question_next
    if '?' in message_text:
        # Split by sentence boundaries (., !, ?)
        sentences = re.split(r'([.!?]+)', message_text)
        non_question_parts = []
        extracted_questions = []
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Check if this sentence contains a question mark
            if '?' in sentence:
                # This is a question - extract it
                question_sentence = sentence
                # Include the punctuation from next item if it exists
                if i + 1 < len(sentences) and sentences[i + 1].strip():
                    question_sentence += sentences[i + 1]
                    i += 1
                extracted_questions.append(question_sentence.strip())
            else:
                # This is a statement - keep it
                if sentence.strip():
                    non_question_parts.append(sentence)
            i += 1
        
        # Reconstruct message without questions
        cleaned_message = ''.join(non_question_parts).strip()
        
        # If we extracted questions, update the fields
        if extracted_questions:
            # Use cleaned message (or fallback if empty)
            if cleaned_message:
                result.message = cleaned_message
            else:
                # If message becomes empty after removing questions, use a generic empathetic response
                result.message = "I understand. Let me ask you something."
            
            # Combine extracted questions with existing question_next
            all_questions = ' '.join(extracted_questions)
            if question_next_text:
                # If question_next already exists, prefer it (it's likely more specific)
                # But log that we found questions in message
                pass
            else:
                # Use the extracted question(s)
                result.question_next = all_questions

    # Final validation: ensure we have valid content after post-processing
    if not result.message or not result.message.strip():
        logger.warning("⚠️  Message is empty after post-processing, using fallback")
        result.message = "I understand."
    
    if not result.question_next or not result.question_next.strip():
        logger.warning("⚠️  question_next is empty after post-processing, using fallback")
        result.question_next = "How are you feeling today?"
    
    # Update slots
    updated_slots = apply_slot_updates(slots, result.slotUpdates or {})

    # Log output
    write_detailed_log(
        {
            "type": "core_chat_output",
            "timestamp": now_iso(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": result.message,
            "question_next": result.question_next,
            "slots_after": updated_slots,
            "risk_assessment": result.riskAssessment,
            "next_steps": result.nextSteps,
        },
        username=str(user_id) if user_id is not None else None,
        conversation_id=str(conversation_id) if conversation_id is not None else None,
    )

    return result, updated_slots


