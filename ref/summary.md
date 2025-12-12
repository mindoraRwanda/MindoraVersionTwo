# Mindora Backend Refactoring Summary

## Overview

This document summarizes the comprehensive refactoring of the Mindora backend application to align with the reference app architecture (`ref/app/`), focusing on simplicity, maintainability, performance, and adherence to best practices.

## Major Changes

### 1. Structured Logging System (Ref-App Style)

**Location**: `backend/app/utils/logging.py`

- **Added comprehensive structured logging** modeled after `ref/app/utils/logging.py`
- **Features**:
  - JSONL-based logging with automatic experiment/config metadata enrichment
  - Per-user and per-conversation log files (`logs/<username>/conv_<id>_detailed.jsonl`)
  - Automatic attachment of settings metadata (model, temperature, RAG config, etc.)
  - Functions: `write_detailed_log()`, `now_iso()`, `write_conversation_log()`, `save_conversation_snapshot()`
- **Integration**:
  - Integrated into `StatefulMentalHealthPipeline.process_query()` with detailed input/output/error logging
  - Logs include processing time, LLM calls, steps completed, response confidence, and errors

### 2. Chat Pipeline Optimization

**Location**: `backend/app/dependencies.py`, `backend/app/services/stateful_pipeline.py`

- **Single Shared Pipeline Instance**:
  - Pipeline now loads once at application startup via `service_container`
  - Module-level fallback singleton ensures no duplicate pipeline construction
  - Per-request only updates `db` and `background` attributes (no graph recompilation)
- **Performance Impact**:
  - Eliminated repeated LangGraph graph compilation overhead
  - LLM provider and RAG services initialized once and reused
  - Reduced latency for subsequent requests

### 3. Core Chat System Revamp (Ref-App Patterns)

**Location**: `backend/app/services/core_chat.py`, `backend/app/services/diagnostic_slots.py`

#### Diagnostic Slots System

- **New Module**: `backend/app/services/diagnostic_slots.py`
  - Ported slot management from `ref/app/services/slots.py`
  - Enums: `YesNoUnknown`, `AppetiteDelta`, `SocialSupportLevel`, `Stressor`
  - Functions: `get_default_slots()`, `apply_slot_updates()`, coercion utilities
  - Tracks: sleep issues, appetite changes, concentration, social support, stressors, exercise, sleep hours

#### Structured Core Chat Engine

- **New Module**: `backend/app/services/core_chat.py`
  - `AssistantTurnState` Pydantic model matching ref app's structured output schema
  - `build_system_rules()`: Creates system prompt with ref app's contract:
    - One diagnostic question per turn
    - ≤2 short sentences for responses
    - Strict JSON output with required `slotUpdates`
    - Last-turn-wins semantics for slot updates
  - `run_core_chat_turn()`: Executes structured chat with:
    - Diagnostic slots integration
    - Emotion and strategy hints from LangGraph
    - RAG/knowledge context
    - Conversation history
    - Detailed logging (input/output)

#### Pipeline State Extensions

- **Updated**: `backend/app/services/pipeline_state.py`
  - Added `diagnostic_slots: Dict[str, Any]` to `StatefulPipelineState`
  - Added `assistant_structured_output: Optional[Dict[str, Any]]` for full structured turn
  - `create_initial_pipeline_state()` now initializes slots via `get_default_slots()`

#### GenerateResponseNode Refactoring

- **Updated**: `backend/app/services/pipeline_nodes.py`
  - `GenerateResponseNode.execute()` now uses structured core chat instead of ad-hoc text generation
  - Preserves all existing upstream behavior (validation, crisis detection, emotion, evaluation, RAG, cultural context)
  - Calls `run_core_chat_turn()` with emotion/strategy hints and RAG context
  - Updates state with: `generated_content` (message), `diagnostic_slots` (updated), `assistant_structured_output` (full turn)
  - Maintains backward compatibility: existing API endpoints continue to work

### 4. Knowledge Base System Migration (Ref-App Style)

**Location**: `backend/app/services/kb.py`, `backend/app/services/rag_enhancement_node.py`

#### New KB Service

- **New Module**: `backend/app/services/kb.py` (ported from `ref/app/services/kb.py`)
  - Loads KB cards from `kb/cards/*.jsonl` files (configurable via `KB_DIR` setting)
  - TF-IDF vectorization over card fields: `title`, `tags`, `when_to_use`, `bot_say`, `steps`
  - Optional local Qdrant integration (when `QDRANT_LOCAL_PATH` and embeddings available)
  - Functions:
    - `load_kb_cards()`: Loads all cards from JSONL files
    - `initialize_kb()`: Fits TF-IDF and optionally sets up Qdrant
    - `retrieve_kb()`: TF-IDF-based retrieval
    - `retrieve_kb_semantic()`: Qdrant-based semantic search (fallback)
    - `retrieve_kb_hybrid()`: Hybrid retrieval with detailed logging
  - Logging: All retrievals logged via `write_detailed_log()` with `type="kb_retrieval"`

#### RAG Enhancement Node Update

- **Updated**: `backend/app/services/rag_enhancement_node.py`
  - Replaced `UnifiedRAGService` dependency with KB module
  - `_retrieve_knowledge()` now uses `retrieve_kb_hybrid()` instead of vector DB search
  - Adapts KB cards to existing format for backward compatibility
  - All filtering, ranking, and context formatting logic preserved

#### Pipeline Integration

- **Updated**: `backend/app/services/stateful_pipeline.py`
  - Always constructs KB-backed `RAGEnhancementNode`
  - Routing: `query_evaluation` → `rag_enhancement` → strategy nodes (empathy/elaboration/etc.)

#### Application Startup

- **Updated**: `backend/app/main.py`
  - Calls `initialize_kb()` during FastAPI startup (after service initialization)
  - Pre-loads and fits TF-IDF over KB cards for fast retrieval

### 5. Service Container Updates

**Location**: `backend/app/services/service_container.py`

- **Removed Dependencies**:
  - `llm_service` no longer depends on `unified_rag_service` (KB handled in pipeline)
  - `stateful_pipeline` no longer depends on `unified_rag_service` (uses KB-backed node)
- **Legacy Code Cleanup**:
  - Commented out `UnifiedRAGService` imports and factory methods
  - Removed RAG service injection into LLM service (no longer needed for core chat)

### 6. LLM Service Updates

**Location**: `backend/app/services/llm_service.py`

- **Removed RAG Dependencies**:
  - Removed `UnifiedRAGService` import and usage
  - Replaced RAG retrieval block with no-op (retrieval now handled by KB-backed pipeline node)
  - Removed RAG service initialization test
  - `set_rag_service()` marked as legacy (kept for compatibility, unused in core flow)

### 7. File Cleanup - Removed Legacy Vector DB System

#### Deleted Files:

1. **`backend/app/services/unified_rag_service.py`**

   - Old vector database service (Qdrant + SentenceTransformer)
   - Replaced by KB-card based retrieval

2. **`backend/app/utils/vector_db_manager.py`**

   - Legacy vector DB management utility
   - No longer needed with KB system

3. **`backend/app/utils/vector_db_populator.py`**

   - Legacy vector DB population utility
   - Replaced by KB card JSONL files

4. **`populate_vector_db.py`** (root level)

   - Top-level vector DB population script
   - Superseded by KB card system

5. **`query_vector_db.py`** (root level)

   - Legacy query utility for vector DB
   - No longer needed

6. **`backend/app/services/stateful_pipeline backup.py`**
   - Obsolete backup implementation
   - Removed to simplify codebase

#### Updated Files:

- **`backend/app/utils/__init__.py`**: Removed exports for deleted vector DB utilities

## Architecture Improvements

### Three-Way Routing (Preserved & Enhanced)

The existing three-way routing system remains intact and is now enhanced with structured output:

1. **General Filter** (`QueryValidationNode`):

   - Routes random queries → END
   - Routes greetings/casual → direct to `generate_response`
   - Routes mental health queries → crisis + therapeutic branches

2. **Crisis Mode** (`CrisisDetectionNode` → `CrisisAlertNode`):

   - Detects crisis with severity classification
   - Logs crisis and notifies therapists
   - Generates culturally appropriate crisis responses

3. **Therapeutic Flow** (`EmotionDetectionNode` → `QueryEvaluationNode` → Strategy Nodes):
   - Emotion detection with youth-specific patterns
   - Strategy selection (empathy, elaboration, suggestion, guidance, idle)
   - **NEW**: Now uses structured core chat with diagnostic slots and risk assessment

### LangGraph Utilities (Preserved)

All existing LangGraph components remain functional:

- `pipeline_state.py`: State management with new slot fields
- `pipeline_nodes.py`: All nodes preserved, `GenerateResponseNode` enhanced
- `rag_enhancement_node.py`: Now KB-backed instead of vector DB
- Validator, crisis detection, emotion detection, query evaluation all intact

## Backward Compatibility

### API Endpoints (Unchanged)

- `/auth/messages` (POST): Still returns `{"id", "sender", "content", "timestamp"}`
- `/voice/messages` (POST): Still returns message + optional audio
- All existing response shapes preserved

### Additional Data (Non-Breaking)

Pipeline results now include **additional** fields (existing fields unchanged):

- `diagnostic_slots`: Current slot state
- `assistant_structured_output`: Full structured turn (message, slots, risk, next steps)

### Existing Features (Preserved)

- Crisis detection and alerting
- Emotion classification
- Cultural context integration
- Safety filtering
- Session state management
- All database models and relationships

## Configuration

### Settings Integration

- KB directory: `KB_DIR` (defaults to `kb/cards`)
- Optional Qdrant: `QDRANT_LOCAL_PATH` for semantic fallback
- Embedding model: `EMB_MODEL_NAME` (defaults to `sentence-transformers/all-MiniLM-L6-v2`)
- All existing settings remain functional

### Environment Variables

- `KB_DIR`: Path to KB cards directory
- `QDRANT_LOCAL_PATH`: Optional local Qdrant path for semantic search
- `EMB_MODEL_NAME`: Embedding model for semantic search (if enabled)
- All existing LLM, database, and safety settings unchanged

## Testing & Validation

### Linter Status

- All modified files pass linting checks
- No syntax or import errors introduced

### Test Compatibility

- Existing test structure preserved
- Test files reference updated imports where necessary
- No breaking changes to test interfaces

## Performance Improvements

1. **Pipeline Initialization**: Single shared instance eliminates repeated graph compilation
2. **KB Retrieval**: TF-IDF is faster than vector DB for small-to-medium card sets
3. **Structured Output**: More efficient than free-form text generation (schema-enforced)
4. **Logging**: Non-blocking, structured logs enable better observability

## Migration Notes

### For Developers

- KB cards should be placed in `kb/cards/*.jsonl` format
- Each card should have: `title`, `tags`, `when_to_use`, `bot_say`, `steps` fields
- KB is automatically initialized on application startup
- No manual vector DB population required

### For Operations

- Old vector DB collections can be safely removed
- KB cards are version-controlled and easier to manage than vector embeddings
- Logs are now structured and easier to analyze

## Summary of Benefits

1. **Simplicity**: Removed complex vector DB setup, replaced with simple JSONL-based KB
2. **Maintainability**: Codebase aligned with reference app patterns, easier to navigate
3. **Performance**: Single pipeline instance, faster KB retrieval, structured output
4. **Observability**: Comprehensive structured logging throughout
5. **Extensibility**: Diagnostic slots and structured output enable richer features
6. **Compatibility**: All existing APIs and features preserved, non-breaking changes

## Files Modified

### New Files

- `backend/app/services/kb.py`
- `backend/app/services/core_chat.py`
- `backend/app/services/diagnostic_slots.py`
- `backend/app/utils/logging.py`

### Modified Files

- `backend/app/services/stateful_pipeline.py`
- `backend/app/services/pipeline_state.py`
- `backend/app/services/pipeline_nodes.py`
- `backend/app/services/rag_enhancement_node.py`
- `backend/app/services/service_container.py`
- `backend/app/services/llm_service.py`
- `backend/app/dependencies.py`
- `backend/app/main.py`
- `backend/app/utils/__init__.py`

### Deleted Files

- `backend/app/services/unified_rag_service.py`
- `backend/app/services/stateful_pipeline backup.py`
- `backend/app/utils/vector_db_manager.py`
- `backend/app/utils/vector_db_populator.py`
- `populate_vector_db.py`
- `query_vector_db.py`

## Next Steps (Optional Enhancements)

1. **Persistence**: Store `diagnostic_slots` and `assistant_structured_output` in database for analytics
2. **API Extensions**: Add endpoints to expose structured output and slot history
3. **Frontend Integration**: Update frontend to consume structured output for richer UI
4. **Analytics**: Leverage structured logs for risk pattern analysis and slot trajectory tracking
