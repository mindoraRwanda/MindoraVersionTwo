from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.app.services.llm_service import LLMService
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from datetime import datetime
import json

# Import service container for dependency injection
from backend.app.services.service_container import get_service
from backend.app.services.langgraph_state import QueryType
from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal

# Create a router
router = APIRouter(prefix="/api", tags=["chat"])

# Dependency injection functions using service container
def get_llm_service():
    """Get LLM service from service container"""
    return get_service("llm_service")

def get_query_validator():
    """Get query validator from service container"""
    return get_service("query_validator")

def get_session_manager():
    """Get session manager from service container"""
    return get_service("session_manager")

def get_crisis_interceptor():
    """Get crisis interceptor from service container"""
    return get_service("crisis_interceptor")

def get_state_router():
    """Get state router from service container"""
    return get_service("state_router")

def get_langgraph_state_router():
    """Get LangGraph state router from service container"""
    return get_service("langgraph_state_router")

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
    llm_service = Depends(get_llm_service),
    query_validator = Depends(get_query_validator),
    current_user = Depends(get_current_user)
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
    should_proceed = False
    is_crisis = False
    query_type = "unclear"
    routing_decision = "standard_processing"

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
            # Use new stateful conversation system for mental health queries
            if is_crisis:
                # Crisis situation - use crisis interceptor
                response_text = await handle_crisis_query(message.message, validation_result)
            else:
                # Mental health query - use stateful conversation system
                response_data = await handle_stateful_conversation(
                    conversation_id, message.message, current_user.id, validation_result, conversation_history=conversation
                )
                response_text = response_data.get("response", "I'm here to support you.")
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
            conversation_history=conversation,
            user_gender=current_user.gender
        )

    # Add assistant response to conversation
    timestamp = datetime.now().isoformat()

    # Include LLM-enhanced conversation metadata if available
    llm_enhanced_metadata = None
    if should_proceed and not is_crisis:
        # Check if this was an LLM-enhanced conversation (response_data would be defined)
        try:
            response_data = locals().get('response_data')
            if response_data:
                llm_enhanced_metadata = {
                    "conversation_state": response_data.get("current_state", "unknown"),
                    "response_type": response_data.get("response_type", "unknown"),
                    "next_state": response_data.get("next_state", "unknown"),
                    "confidence": response_data.get("confidence", 0.0),
                    "llm_reasoning": response_data.get("llm_reasoning", ""),
                    "cultural_considerations": response_data.get("cultural_considerations", {}),
                    "suggested_actions": response_data.get("suggested_actions", []),
                    "llm_enhanced": True
                }
        except (NameError, KeyError):
            pass  # response_data not available, use None

    assistant_message = {
        "role": "assistant",
        "text": response_text,
        "timestamp": timestamp,
        "query_validation": {
            "query_type": validation_result.get("query_type", "unknown") if validation_result else "unknown",
            "confidence": validation_result.get("confidence", 0.0) if validation_result else 0.0,
            "routing_decision": validation_result.get("routing_decision", "unknown") if validation_result else "unknown"
        } if validation_result else None,
        "llm_enhanced_conversation": llm_enhanced_metadata
    }
    conversation.append(assistant_message)

    # Update conversation store
    conversation_store[conversation_id] = conversation

    return ChatResponse(
        response=response_text,
        timestamp=timestamp,
        conversation_id=conversation_id
    )


async def handle_stateful_conversation(conversation_id: str, user_message: str,
                                       user_id: str, validation_result: Dict[str, Any],
                                       conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Handle conversation using the LLM-enhanced stateful conversation system

    Args:
        conversation_id: Unique conversation identifier
        user_message: User's message
        user_id: User identifier
        validation_result: Query validation results
        conversation_history: Current conversation history for context

    Returns:
        Response data from LLM-enhanced stateful conversation system
    """
    try:
        # Use LLM-enhanced router for intelligent state decisions
        langgraph_router = get_langgraph_state_router()

        # Get current session to merge conversation history
        session_mgr = get_session_manager()
        session = session_mgr.get_session(conversation_id)

        # Update session with current conversation history for consistency
        if conversation_history:
            # Update session manager with the complete conversation history
            for message in conversation_history:
                if message["role"] == "user":
                    session_mgr.add_message_to_history(conversation_id, "user", message.get("text", message.get("content", "")))
                elif message["role"] == "assistant":
                    session_mgr.add_message_to_history(conversation_id, "assistant", message.get("text", message.get("content", "")))

        # Pass conversation context to the router
        response_data = await langgraph_router.route_conversation(
            conversation_id,
            user_message,
            emotion_data=validation_result.get("emotion_detection"),
            cultural_context={"rwandan_context": True},
            query_validation=validation_result
        )

        return response_data

    except Exception as e:
        print(f"Error in LLM-enhanced stateful conversation: {e}")
        try:
            # Fallback to basic state actions
            session_mgr = get_session_manager()
            session = session_mgr.get_session(conversation_id)
            if not session:
                session_mgr.create_session_with_id(conversation_id, user_id)

            # Try to get state actions service
            try:
                from backend.app.services.state_actions import state_actions
                response_data = await state_actions.execute_state_action(
                    conversation_id, user_message
                )
                return response_data
            except ImportError:
                # If state_actions not available, use LLM service directly
                llm_svc = get_llm_service()
                fallback_response = await llm_svc.generate_response(
                    user_message,
                    conversation_history=conversation_history or [],
                    user_gender="unknown"
                )
                return {
                    "response": fallback_response,
                    "current_state": "fallback",
                    "response_type": "fallback"
                }

        except Exception as e2:
            print(f"Error in fallback stateful conversation: {e2}")
            # Final fallback to original LLM service
            llm_svc = get_llm_service()
            fallback_response = await llm_svc.generate_response(
                user_message,
                conversation_history=conversation_history or [],
                user_gender="unknown"
            )
            return {
                "response": fallback_response,
                "current_state": "fallback",
                "response_type": "fallback"
            }


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
        query_validator = Depends(get_query_validator)
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
