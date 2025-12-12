from fastapi import BackgroundTasks
from .services.stateful_pipeline import StatefulMentalHealthPipeline
from .services.service_container import get_service

# Fallback singleton used only if the global service container is not available.
_fallback_pipeline: StatefulMentalHealthPipeline | None = None


def get_stateful_pipeline(db, background: BackgroundTasks) -> StatefulMentalHealthPipeline:
    """
    Get the shared stateful mental health pipeline.

    Primary path: use the globally managed instance from the service container,
    which is initialized once at application startup.

    Fallback path (e.g. for isolated tests or scripts): create a single
    module-level pipeline instance and reuse it across calls so the LangGraph
    graph and model are not recompiled on every request.
    """
    global _fallback_pipeline

    try:
        pipeline: StatefulMentalHealthPipeline = get_service("stateful_pipeline")
    except Exception:
        # Lazily initialize a single fallback instance if the container is not ready.
        if _fallback_pipeline is None:
            from .services.stateful_pipeline import initialize_stateful_pipeline

            try:
                llm_service = get_service("llm_service")
                llm_provider = getattr(llm_service, "llm_provider", None)
            except Exception:
                llm_provider = None

            _fallback_pipeline = initialize_stateful_pipeline(
                llm_provider=llm_provider,
                db=db,
                background=background,
            )

        pipeline = _fallback_pipeline

    # Inject request-scoped resources onto the shared pipeline
    pipeline.db = db
    pipeline.background = background
    return pipeline

