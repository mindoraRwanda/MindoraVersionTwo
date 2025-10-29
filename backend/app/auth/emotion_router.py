# app/routers/emotion_router.py
from fastapi import APIRouter, HTTPException, Depends
from .schemas import AnalysisRequest, EmotionRequest
from ..services.emotion_classifier import LLMEmotionClassifier
from ..services.service_container import get_service
# Legacy chatbot_insights_pipeline removed - using stateful pipeline instead

router = APIRouter()

@router.post("/emotion")
async def detect_emotion(request: EmotionRequest):
    """
    Detect emotion using LLM-powered analysis with cultural context.
    
    This endpoint now uses sophisticated LLM analysis instead of basic keyword matching,
    providing more accurate emotion detection with cultural sensitivity.
    """
    try:
        # Get the LLM emotion classifier service
        emotion_classifier = get_service("emotion_classifier")
        if not emotion_classifier:
            raise HTTPException(
                status_code=503, 
                detail="Emotion classification service not available"
            )
        
        # Use LLM-powered emotion classification
        result = await emotion_classifier.classify_emotion(
            text=request.text,
            user_gender=getattr(request, 'user_gender', None)
        )
        
        return {
            "emotion": result.get("emotion", "neutral"),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "keywords": result.get("keywords", []),
            "intensity": result.get("intensity", "medium"),
            "cultural_context": result.get("cultural_context", {}),
            "analysis_type": "llm_powered"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Emotion classification failed: {str(e)}"
        )

@router.post("/analyze", tags=["Mental Health Analysis"])
def analyze_user_message(request: AnalysisRequest):
    """
    Analyze a user message for emotional and psychological risk indicators.
    Note: Legacy analysis pipeline removed. Use the stateful pipeline via /auth/messages endpoint instead.
    """
    raise HTTPException(
        status_code=410, 
        detail="Legacy analysis endpoint removed. Please use the stateful pipeline via /auth/messages endpoint for comprehensive analysis."
    )

@router.post("/detect/medications", tags=["Utilities"])
def extract_medications(request: AnalysisRequest):
    """
    Legacy medication detection endpoint removed.
    Use the stateful pipeline via /auth/messages endpoint instead.
    """
    raise HTTPException(
        status_code=410, 
        detail="Legacy medication detection endpoint removed. Use the stateful pipeline via /auth/messages endpoint."
    )

@router.post("/detect/suicide-risk", tags=["Utilities"])
def check_suicide_flag(request: AnalysisRequest):
    """
    Legacy suicide risk detection endpoint removed.
    Use the stateful pipeline via /auth/messages endpoint instead.
    """
    raise HTTPException(
        status_code=410, 
        detail="Legacy suicide risk detection endpoint removed. Use the stateful pipeline via /auth/messages endpoint."
    )

@router.post("/reindex", tags=["Admin"])
def rebuild_vector_knowledge():
    """
    Legacy knowledge base reindexing endpoint removed.
    Knowledge base management is now handled by the RAG service.
    """
    raise HTTPException(
        status_code=410, 
        detail="Legacy knowledge base reindexing removed. Knowledge base is now managed by the RAG service."
    )
