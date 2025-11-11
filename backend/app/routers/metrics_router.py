"""
Metrics API Router

Endpoints for retrieving chatbot performance metrics and analytics.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/emotion")
async def get_emotion_metrics():
    """
    Get emotion classification metrics.
    
    Returns:
        - Total classifications
        - Average confidence
        - Average processing time
        - Emotion distribution
        - Cultural marker detection rate
    """
    try:
        # Import pipeline from service container
        from ..services.service_container import service_container
        pipeline = service_container.get_service("stateful_pipeline")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        summary = pipeline.emotion_metrics.get_summary()
        
        return {
            "status": "success",
            "metrics": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving emotion metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline")
async def get_pipeline_metrics():
    """
    Get pipeline execution metrics.
    
    Returns:
        - Total executions
        - Average response time
        - LLM call efficiency
        - Error rates
        - Strategy distribution
    """
    try:
        from ..services.service_container import service_container
        pipeline = service_container.get_service("stateful_pipeline")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        summary = pipeline.pipeline_metrics.get_summary()
        
        return {
            "status": "success",
            "metrics": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_metrics_summary():
    """
    Get combined metrics summary for dashboard display.
    
    Returns comprehensive overview of:
    - Emotion detection performance
    - Pipeline execution stats
    - System health indicators
    """
    try:
        from ..services.service_container import service_container
        pipeline = service_container.get_service("stateful_pipeline")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        emotion_summary = pipeline.emotion_metrics.get_summary()
        pipeline_summary = pipeline.pipeline_metrics.get_summary()
        
        return {
            "status": "success",
            "generated_at": datetime.now().isoformat(),
            "emotion_metrics": emotion_summary,
            "pipeline_metrics": pipeline_summary,
            "system_health": {
                "ml_classifier_active": pipeline.ml_emotion_classifier is not None,
                "rag_service_active": pipeline.rag_service is not None,
                "total_metrics_tracked": (
                    len(pipeline.emotion_metrics.metrics) +
                    len(pipeline.pipeline_metrics.metrics)
                )
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/emotion")
async def export_emotion_metrics():
    """Export emotion metrics to JSON file."""
    try:
        from ..services.service_container import service_container
        pipeline = service_container.get_service("stateful_pipeline")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"metrics_emotion_{timestamp}.json"
        
        pipeline.emotion_metrics.export_json(filepath)
        
        return {
            "status": "success",
            "message": f"Emotion metrics exported to {filepath}",
            "filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error exporting emotion metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/pipeline")
async def export_pipeline_metrics():
    """Export pipeline metrics to JSON file."""
    try:
        from ..services.service_container import service_container
        pipeline = service_container.get_service("stateful_pipeline")
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"metrics_pipeline_{timestamp}.json"
        
        pipeline.pipeline_metrics.export_json(filepath)
        
        return {
            "status": "success",
            "message": f"Pipeline metrics exported to {filepath}",
            "filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error exporting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
