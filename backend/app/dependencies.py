from fastapi import Depends, BackgroundTasks
from sqlalchemy.orm import Session
from .services.stateful_pipeline import StatefulMentalHealthPipeline

def get_stateful_pipeline(db, background: BackgroundTasks):
    """Get stateful mental health pipeline from service container."""
    from .services.service_container import get_service
    try:
        pipeline = get_service("stateful_pipeline")
        pipeline.db = db
        pipeline.background = background
        return pipeline
    except Exception:
        # Fallback to creating a new instance
        from .services.stateful_pipeline import initialize_stateful_pipeline
        llm_service = get_service("llm_service")
        return initialize_stateful_pipeline(llm_provider=llm_service.llm_provider if llm_service else None, db=db, background=background)
