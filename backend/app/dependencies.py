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
        print("✅ Retrieved stateful_pipeline from service container")
        return pipeline
    except Exception as e:
        print(f"⚠️ get_stateful_pipeline fallback: {e}")
        # Fallback to creating a new instance
        from .services.stateful_pipeline import initialize_stateful_pipeline
        llm_service = None
        try:
            llm_service = get_service("llm_service")
        except Exception as inner_e:
            print(f"⚠️ get_stateful_pipeline fallback could not retrieve llm_service: {inner_e}")
        return initialize_stateful_pipeline(llm_provider=llm_service.llm_provider if llm_service else None, db=db, background=background)
