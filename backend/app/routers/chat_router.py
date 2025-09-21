from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Import the LLM service and query validator
from backend.app.services.llm_service_refactored import LLMService
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryType

# Create a router
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize the services (lazy loading)
_llm_service = None
_query_validator = None

def get_llm_service():
    """Lazy initialization of the LLM service"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(use_vllm=False, provider_name="ollama", model_name="gemma3:1b")

    return _llm_service

def get_query_validator():
    """Lazy initialization of the query validator"""
    global _query_validator
    if _query_validator is None:
        llm_service = get_llm_service()
        _query_validator = LangGraphQueryValidator(llm_provider=llm_service.llm_provider)

    return _query_validator

# Define the message schema
class Message(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# Define the response schema
class ChatResponse(BaseModel):
    response: str
    timestamp: str
    conversation_id: Optional[str] = None

# Enhanced conversation storage with query validation
conversation_store = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: Message,
    llm_service: LLMService = Depends(get_llm_service),
    query_validator: LangGraphQueryValidator = Depends(get_query_validator)
):
    """
    Enhanced chat endpoint with query validation.

    This endpoint now uses LangGraph query validation to:
    1. Validate if the query is mental health related
    2. Filter out unrelated questions
    3. Route appropriate queries to the main LLM service
    4. Provide contextual responses for off-topic queries
    """
    conversation_id = message.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    conversation = conversation_store.get(conversation_id, [])

    # Add user message to conversation
    user_message = {
        "role": "user",
        "text": message.message,
        "timestamp": datetime.now().isoformat()
    }
    conversation.append(user_message)

    # Step 1: Execute the LangGraph workflow
    validation_result = None
    try:
        validation_result = await query_validator.execute_workflow(
            message.message,
            conversation_history=conversation
        )

        # Step 2: Check if conversation should proceed
        should_proceed = validation_result.get("should_proceed_to_conversation", False)
        query_type = validation_result.get("query_type", "unclear")
        is_crisis = validation_result.get("is_crisis", False)
        routing_decision = validation_result.get("routing_decision", "standard_processing")

        print(f"Query validation result: {json.dumps(validation_result, indent=2)}")

        # Step 3: Handle different query types based on workflow decision
        if should_proceed:
            # Only proceed to conversation for mental support or crisis queries
            if is_crisis:
                # Crisis situation - route to crisis intervention
                response_text = await handle_crisis_query(message.message, validation_result)
            else:
                # Mental health query - process with main LLM
                response_text = await llm_service.generate_response(
                    message.message,
                    conversation_history=conversation
                )
        else:
            # Do not proceed to conversation - return filtered response
            if query_type == "random_question":
                response_text = await handle_random_question(message.message, validation_result)
            elif query_type == "unclear":
                response_text = await handle_unclear_query(message.message, validation_result)
            else:
                # Fallback for other cases
                response_text = validation_result.get("final_response", "I'm here to help with mental health and emotional support. Could you please clarify what you'd like help with?")

    except Exception as e:
        # Fallback to original behavior if validation fails
        print(f"Query validation failed: {e}")
        response_text = await llm_service.generate_response(
            message.message,
            conversation_history=conversation
        )

    # Add assistant response to conversation
    timestamp = datetime.now().isoformat()
    assistant_message = {
        "role": "assistant",
        "text": response_text,
        "timestamp": timestamp,
        "query_validation": {
            "query_type": validation_result.get("query_type", "unknown") if validation_result else "unknown",
            "confidence": validation_result.get("confidence", 0.0) if validation_result else 0.0,
            "routing_decision": validation_result.get("routing_decision", "unknown") if validation_result else "unknown"
        } if validation_result else None
    }
    conversation.append(assistant_message)

    # Update conversation store
    conversation_store[conversation_id] = conversation

    return ChatResponse(
        response=response_text,
        timestamp=timestamp,
        conversation_id=conversation_id
    )


async def handle_crisis_query(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle crisis queries with appropriate intervention.

    Args:
        query: The crisis query
        validation_result: Validation results from LangGraph

    Returns:
        Crisis intervention response
    """
    crisis_severity = validation_result.get("crisis_severity", "medium")

    if crisis_severity == "critical":
        return (
            "🚨 I detect this may be a crisis situation. Your safety is the highest priority.\n\n"
            "Please reach out for immediate professional help:\n"
            "• Emergency Services: 112\n"
            "• Mental Health Helpline: 114 (24/7, free, confidential)\n"
            "• Ndera Neuropsychiatric Hospital: +250 781 447 928\n\n"
            "You don't have to face this alone. Professional crisis support is essential right now. "
            "Please contact emergency services immediately."
        )
    else:
        return (
            "I'm concerned about what you're sharing. While this may not be an immediate crisis, "
            "it's important to talk to a professional.\n\n"
            "Available resources:\n"
            "• Mental Health Helpline: 114\n"
            "• Emergency Services: 112\n"
            "• Local health centers in your area\n\n"
            "Would you like me to help you connect with these services?"
        )
    
    
    @router.post("/validate-query")
    async def validate_query(
        message: Message,
        query_validator: LangGraphQueryValidator = Depends(get_query_validator)
    ):
        """
        Endpoint to validate a query without generating a full response.
    
        This endpoint allows clients to check if a query is mental health related
        before sending it to the main chat endpoint.
        """
        try:
            validation_result = await query_validator.execute_workflow(message.message)
    
            return {
                "query": message.message,
                "validation": validation_result,
                "is_mental_health_related": validation_result.get("query_type") == "mental_support",
                "is_crisis": validation_result.get("is_crisis", False),
                "should_process": validation_result.get("should_proceed_to_conversation", False),
                "confidence": validation_result.get("confidence", 0.0)
            }
    
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Query validation failed: {str(e)}"
            )
    
    
    @router.get("/chat-stats")
    async def get_chat_stats():
        """
        Get statistics about chat conversations and query validation.
        """
        total_conversations = len(conversation_store)
        total_messages = sum(len(conv) for conv in conversation_store.values())
    
        # Calculate validation stats (mock data for now)
        validation_stats = {
            "mental_health_queries": 0,
            "random_questions": 0,
            "crisis_queries": 0,
            "unclear_queries": 0,
            "total_validated": 0
        }
    
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "validation_stats": validation_stats,
            "system_status": "operational"
        }


async def handle_random_question(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle random questions with filtered responses.

    Args:
        query: The random question
        validation_result: Validation results from LangGraph

    Returns:
        Filtered response for random questions
    """
    suggestions = validation_result.get("suggestions", [])

    # Check if this is a technical/programming question
    technical_keywords = [
        'python', 'javascript', 'java', 'programming', 'coding', 'software',
        'computer', 'install', 'setup', 'configuration', 'bug', 'error',
        'debug', 'code', 'script', 'algorithm', 'function', 'variable',
        'class', 'method', 'api', 'framework', 'library', 'package'
    ]

    is_technical = any(keyword in query.lower() for keyword in technical_keywords)

    if is_technical:
        return (
            "I notice you're asking about a technical or programming topic. "
            "While I'm here to support your mental health and emotional well-being, "
            "I'm not designed to provide technical assistance or programming help.\n\n"
            f"Your question: '{query}'\n\n"
            "For technical questions, I recommend:\n"
            "• Consulting official documentation\n"
            "• Using search engines like Google\n"
            "• Asking in programming communities (Stack Overflow, Reddit)\n"
            "• Using AI assistants designed for coding\n\n"
            "However, if this technical issue is causing you stress or affecting your mental health, "
            "I'm here to help you cope with those feelings. Would you like to talk about how this is impacting you emotionally?"
        )
    else:
        return (
            "I understand you have a question, but I'm primarily designed to support mental health and emotional well-being. "
            "While I can try to help with general questions, my expertise is in providing mental health support.\n\n"
            f"Your question: '{query}'\n\n"
            "If this is related to your mental health or emotional well-being, please let me know how I can support you. "
            "For other topics, you might want to consult a general AI assistant or search engine.\n\n"
            "Is there anything related to your mental health or well-being you'd like to discuss?"
        )


async def handle_unclear_query(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle unclear queries by asking for clarification.

    Args:
        query: The unclear query
        validation_result: Validation results from LangGraph

    Returns:
        Clarification request response
    """
    return (
        "I'm not sure I understand your query. Could you please clarify what you'd like help with?\n\n"
        "I'm here to support your mental health and emotional well-being. You can ask me about:\n"
        "• How you're feeling emotionally\n"
        "• Stress or anxiety you're experiencing\n"
        "• Depression or mood concerns\n"
        "• Relationship difficulties\n"
        "• Work or school-related stress\n"
        "• General mental health questions\n\n"
        "What would be most helpful for you right now?"
    )
