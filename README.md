# Mindora Conversational Therapy Chatbot

A comprehensive mental health support platform designed for Rwandan youth, featuring advanced AI capabilities, gender-aware cultural sensitivity, and natural conversational responses. Built with modern web technologies and multiple AI integrations for scalable, reliable mental health support that feels like talking to a culturally aware Rwandan brother or sister.

---

## Features

- **Multi-Provider AI Support**: Integration with Ollama, OpenAI, Groq, and HuggingFace (default: SmolLM3-3B) for flexible LLM deployment
- **Gender-Aware Cultural Context**: Personalized responses based on user gender with natural Kinyarwanda addressing ("murumuna" for brothers, "murumuna wanjye" for sisters)
- **Natural Conversational Responses**: AI speaks like a culturally aware Rwandan brother/sister rather than a clinical assistant
- **Advanced RAG System**: ChromaDB/Qdrant vector database with intelligent document retrieval
- **Emotion Detection**: Real-time emotion classification with gender-aware response adaptation
- **Safety & Ethics**: Comprehensive safety checks, crisis detection, and content filtering
- **Modern Architecture**: FastAPI backend with React frontend, PostgreSQL database
- **LangGraph Integration**: Advanced query validation and conversation flow management
- **Comprehensive Testing**: Unit and integration tests with CI/CD pipeline
- **Docker Deployment**: Containerized services with Docker Compose orchestration
- **GPU Acceleration**: vLLM support for high-performance inference

---

## Prerequisites

- **Docker & Docker Compose** (recommended) - For containerized deployment
- **Python 3.10+** - For local development
- **Node.js v18+** - For frontend development
- **Git** - Version control
- **GPU** (optional) - For vLLM acceleration

### Optional Services
- **Ollama** - Local LLM serving (if using Ollama provider)
- **OpenAI API Key** - For OpenAI provider
- **Groq API Key** - For Groq provider
- **HuggingFace** - Local model support (if using HuggingFace provider)

---

## Quick Start (Docker Compose)

### Step 1: Clone the Repository

```bash
git clone https://github.com/mindoraRwanda/MindoraVersionTwo
cd mindora-chatbot
```

### Step 2: Start All Services

```bash
# Start all services (PostgreSQL, Qdrant, vLLM, Backend, Frontend)
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### Step 3: Access the Application

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Step 4: Configure LLM Provider (Optional)

By default, the system uses HuggingFace with SmolLM3-3B model for optimal privacy and cultural sensitivity. To use other providers:

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export GROQ_API_KEY="your-groq-key"

# Or edit docker-compose.yml and restart
docker-compose down
docker-compose up --build
```

## Manual Setup (Development)

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional packages for RAG
pip install langchain-community qdrant-client chromadb

# Install HuggingFace support (optional)
pip install transformers torch accelerate

# Set up environment variables using the new FastAPI settings system
# The system automatically loads environment-specific configuration from .env files
# Copy the appropriate environment file:
cp .env.example .env
# Edit .env with your API keys and configuration

# Run database migrations
python -c "from backend.app.db.database import engine; from backend.app.db import models; models.Base.metadata.create_all(bind=engine)"

# Start backend
python3 backend/app/db/run_migrations.py
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd test-frontend
npm install
npm run dev
```

### Vector Database Setup

```bash
# Qdrant (via Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or ChromaDB (Python)
pip install chromadb
```

### HuggingFace Model Setup (Default Provider)

```bash
# Install required packages
pip install transformers torch accelerate

# Download the default model (SmolLM3-3B) - automatically handled by the system
# The system will automatically download SmolLM3-3B on first use

# For custom model configuration
export HUGGINGFACE_MODEL_PATH="HuggingFaceTB/SmolLM3-3B"  # Default: SmolLM3-3B
export HUGGINGFACE_DEVICE="cuda"  # Use "cpu" for CPU-only systems

# Test model loading
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM3-3B'); print('Model loaded successfully')"
```

### Document Ingestion

```bash
# Place your mental health documents in backend/mental_health_knowledge/ directory
# Then run the RAG service
python -m backend.app.services.rag_service
```

---

## API Endpoints

### Core Endpoints

#### Authentication
- **POST** `/signup` - User registration
- **POST** `/login` - User login

#### Conversations
- **GET** `/conversations` - List user conversations
- **POST** `/conversations` - Create new conversation
- **DELETE** `/conversations/{id}` - Delete conversation

#### Messages
- **GET** `/conversations/{conversation_id}/messages` - Get messages in conversation
- **POST** `/messages` - Send message
- **GET** `/context` - Get context window

#### Mental Health Insights
- **POST** `/emotion` - Detect emotion in text
- **POST** `/analyze` - Analyze user message
- **POST** `/detect/medications` - Extract medications from text
- **POST** `/detect/suicide-risk` - Check suicide risk flags
- **POST** `/reindex` - Rebuild vector knowledge base

### Example Request

```bash
curl -X POST "http://localhost:8000/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does Rwanda'\''s mental health policy say about community-based care?",
    "conversation_id": "test1",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "response": "Based on Rwanda's mental health policy...",
  "emotion": "curious",
  "crisis_detected": false,
  "sources": ["Rwanda Mental Health Policy 2020"]
}
```

---

## Architecture

### Backend Architecture

The backend follows a modular architecture with clear separation of concerns and a comprehensive FastAPI settings system:

```
backend/
├── __init__.py                    # Backend package initialization
├── main.py                        # FastAPI application entry point
├── Dockerfile                     # Container configuration
├── app/                           # Main application package
│   ├── __init__.py               # App package initialization
│   ├── main.py                   # FastAPI app instance & routes
│   ├── settings/                 # FastAPI settings system (NEW)
│   │   ├── __init__.py          # Settings entry point
│   │   ├── base.py              # Base settings class
│   │   ├── settings.py          # Main settings aggregation
│   │   ├── model.py             # LLM model configuration
│   │   ├── performance.py       # Performance settings
│   │   ├── safety.py            # Safety and content filtering
│   │   ├── cultural.py          # Rwanda-specific cultural context
│   │   ├── database.py          # Database configuration
│   │   ├── emotional.py         # Emotional response settings
│   │   ├── qdrant.py            # Qdrant vector database settings
│   │   └── .env.*               # Environment-specific configuration files
│   ├── auth/                     # Authentication system
│   │   ├── __init__.py
│   │   ├── auth_router.py        # Authentication endpoints
│   │   ├── emotion_router.py     # Emotion analysis endpoints
│   │   ├── schemas.py            # Pydantic schemas
│   │   └── utils.py              # Auth utilities
│   ├── db/                       # Database layer
│   │   ├── __init__.py
│   │   ├── database.py           # Database connection & config
│   │   └── models.py             # SQLAlchemy models
│   ├── models/                   # Additional data models
│   │   └── __init__.py
│   ├── prompts/                  # AI prompt templates
│   │   ├── __init__.py
│   │   ├── cultural_context_prompts.py
│   │   ├── query_classification_prompts.py
│   │   ├── response_approach_prompts.py
│   │   ├── safety_prompts.py
│   │   └── system_prompts.py
│   ├── routers/                  # API route handlers
│   │   ├── __init__.py
│   │   ├── conversations_router.py
│   │   └── messages_router.py
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── crisis_alert_service.py
│   │   ├── emotion_classifier.py
│   │   ├── llm_cultural_context.py
│   │   ├── llm_database_operations.py
│   │   ├── llm_model_manager.py
│   │   ├── llm_providers.py
│   │   ├── llm_safety.py
│   │   ├── llm_service.py
│   │   ├── pipeline_nodes.py
│   │   ├── pipeline_state.py
│   │   ├── rag_enhancement_node.py
│   │   ├── service_container.py
│   │   ├── session_state_manager.py
│   │   ├── stateful_pipeline.py
│   │   └── unified_rag_service.py
│   │   └── mental_health_knowledge/  # Vector DB storage
│   └── utils/                    # Utility functions
│       └── __init__.py
├── requirements.txt              # Python dependencies
└── mental_health_knowledge/      # Vector database files
```

### Component Descriptions

#### **Core Application** (`app/`)
- **`main.py`**: FastAPI application instance with lifespan management, CORS configuration, and route registration
- **`settings/`**: **NEW** Comprehensive FastAPI settings system using Pydantic BaseSettings for type-safe configuration management
- **`auth/`**: Complete authentication system with JWT tokens, password hashing, and user management
- **`db/`**: Database configuration, connection management, and SQLAlchemy models for users, conversations, and messages
- **`models/`**: Additional data models and Pydantic schemas for API contracts

#### **Settings System** (`settings/`)
- **`settings.py`**: Main settings aggregation with environment-specific configuration loading
- **`base.py`**: Base settings class with common functionality for all configuration categories
- **`model.py`**: LLM model configuration, providers, API keys, and model parameters
- **`performance.py`**: Performance settings including timeouts, limits, and RAG configuration
- **`safety.py`**: Content filtering, crisis detection keywords, and safety patterns
- **`cultural.py`**: Rwanda-specific cultural context, resources, and system prompts
- **`database.py`**: Database connection settings, pool configuration, and Redis settings
- **`emotional.py`**: Emotion-specific response guidance and topic adjustments
- **`qdrant.py`**: Qdrant vector database configuration
- **`.env.*`**: Environment-specific configuration files (development, testing, production)

#### **AI Services** (`services/`)
- **`llm_service.py`**: Main orchestrator for LLM operations with full pipeline processing (renamed from refactored version)
- **`llm_providers.py`**: Factory pattern for multiple LLM providers (Ollama, OpenAI, Groq, vLLM)
- **`unified_rag_service.py`**: Unified RAG service combining document processing and retrieval with vector similarity search
- **`emotion_classifier.py`**: Real-time emotion detection and classification
- **`llm_safety.py`**: Content filtering and safety checks
- **`llm_cultural_context.py`**: Gender-aware Rwandan cultural context with natural Kinyarwanda addressing
- **`service_container.py`**: Dependency injection system for service management
- **`stateful_pipeline.py`**: LangGraph-based conversation flow management

#### **Prompt Management** (`prompts/`)
- **`system_prompts.py`**: Core system prompts with cultural context
- **`safety_prompts.py`**: Safety and ethical response templates
- **`cultural_context_prompts.py`**: Rwanda-specific cultural prompts
- **`response_approach_prompts.py`**: Emotion-based response strategies

#### **API Layer** (`routers/`)
- **`conversations_router.py`**: Conversation management and history
- **`messages_router.py`**: Message CRUD operations

#### **Data Storage**
- **Vector Databases**: Chroma/Qdrant for semantic search and document retrieval
- **Relational Database**: PostgreSQL for user data, conversations, and messages
- **Mental Health Knowledge**: Curated documents on Rwandan mental health policies and practices

### Frontend Architecture

```
test-frontend/
├── src/
│   ├── api/                      # API client configuration
│   ├── pages/                    # React components
│   │   ├── ChatDashboard.jsx     # Main chat interface
│   │   ├── WelcomeScreen.jsx     # Landing page
│   │   ├── Sidebar.jsx           # Navigation
│   │   └── useChatAPI.js         # Chat API hooks
│   └── App.js                    # Root component
├── package.json                  # Node.js dependencies
└── tailwind.config.js            # CSS framework config
```

### Testing Architecture

```
tests/
├── unit/                         # Unit tests
│   ├── test_query_validator.py
│   ├── test_prompts.py
│   └── test_langgraph_validator.py
├── integration/                  # Integration tests
│   ├── test_chat_router_integration.py
│   └── test_langgraph_workflow.py
├── conftest.py                   # Pytest fixtures
└── requirements.txt              # Test dependencies
```

### LLM Providers

The system supports multiple LLM providers for maximum flexibility:

#### 1. Ollama (Local)
- **Best for**: Privacy, offline usage, custom models
- **Setup**: `pip install langchain-ollama`
- **Models**: Llama 3, Gemma, local fine-tuned models
- **Configuration**: Set `OLLAMA_BASE_URL` environment variable

#### 2. OpenAI (Cloud)
- **Best for**: High-quality responses, GPT models
- **Setup**: Set `OPENAI_API_KEY` environment variable
- **Models**: GPT-4, GPT-3.5-turbo
- **Cost**: Pay-per-token usage

#### 3. Groq (Cloud)
- **Best for**: Speed, open-source models
- **Setup**: Set `GROQ_API_KEY` environment variable
- **Models**: Llama 2, Mixtral, open-source models
- **Cost**: Free tier available

#### 4. HuggingFace (Local Models)
- **Best for**: Privacy, custom models, offline usage, research models
- **Setup**: `pip install transformers torch accelerate`
- **Default Model**: SmolLM3-3B (HuggingFaceTB/SmolLM3-3B)
- **Models**: Any HuggingFace model (Llama, Mistral, Falcon, custom fine-tuned)
- **Hardware**: CPU or GPU (GPU recommended for larger models)
- **Privacy**: Complete data privacy, models run locally
- **Cost**: Free (no API costs)
- **Flexibility**: Support for any HuggingFace-compatible model

#### 5. vLLM (GPU Acceleration)
- **Best for**: High-throughput inference
- **Setup**: Docker container with GPU support
- **Models**: Optimized for GPU inference
- **Performance**: 10-100x faster than CPU

### Data Sources

**Total Collected:** 140+ documents

**Sources Include:**

- Rwanda Ministry of Health policies and guidelines
- Rwanda Biomedical Center mental health resources
- WHO mhGAP and Mental Health Atlas
- APA and SAMHSA clinical guidelines
- PTSD, PHQ-9, GAD-7 screening tools
- Academic research on mental health in Africa
- Rwanda-specific cultural and contextual resources

### Testing Framework

Comprehensive test suite with:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflow testing
- **Test Categories**: LLM, database, async, performance tests
- **CI/CD**: Automated testing on GitHub Actions
- **Coverage**: HTML coverage reports

Run tests:
```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=backend --cov-report=html tests/
```

---

## Adding More Documents

1. Place PDF documents in `backend/mental_health_knowledge/` directory (where vector databases are stored)
2. Run the document ingestion service:

```bash
python -m backend.app.services.rag_service
```

3. Documents will be automatically:
    - Parsed and chunked
    - Embedded using sentence transformers
    - Stored in vector database (ChromaDB/Qdrant)
    - Made available for RAG queries

**Note**: The system uses the existing vector databases in `backend/mental_health_knowledge/` for document storage and retrieval.

---

## Project Status

### ✅ Current Active Components
- **LLM Service**: `llm_service.py` (main orchestrator with gender-aware responses)
- **Query Validation**: LangGraph-based validation system with emotion detection
- **Vector Storage**: `backend/mental_health_knowledge/` (active vector DB location)
- **Testing**: Comprehensive test suite in `/tests/` directory
- **Settings System**: **NEW** FastAPI settings system with Pydantic BaseSettings for type-safe configuration
- **Cultural Context**: Gender-aware Rwandan cultural integration with natural Kinyarwanda addressing

### ⚠️ Legacy Code (Safe to Remove)
- **`test_*.py` in backend root**: Functionality moved to `/tests/` directory
- **`backend/.env`**: Environment config should be in project root
- **`backend/datasources/`**: Unused directory (use `backend/mental_health_knowledge/` instead)

### 🔧 Recent Updates
- **NEW FastAPI Settings System**: Comprehensive type-safe configuration system
  - Environment-specific configuration files (development, testing, production)
  - Hierarchical configuration categories with validation
  - Secure environment variable management with `MINDORA_` prefix
- **Gender-Aware Cultural Context**: Added personalized responses based on user gender with Kinyarwanda addressing
- **Natural Conversational Responses**: Enhanced AI to speak like a culturally aware Rwandan brother/sister
- **HuggingFace Default Provider**: Set SmolLM3-3B as default model for optimal privacy and cultural sensitivity
- **Enhanced Emotion Detection**: Integrated emotion analysis into LangGraph workflow for better response personalization
- **Service Container**: Implemented dependency injection system for better service management
- **Updated Architecture Documentation**: Comprehensive updates to reflect current structure and capabilities

---

## Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Reset Docker containers
docker-compose down
docker system prune -a
docker-compose up --build
```

#### Database Issues
```bash
# Reset PostgreSQL data
docker-compose down
docker volume rm mindora_postgres_data
docker-compose up postgres
# Migration commands
# Run all migrations
python3 backend/app/db/run_migrations.py

# Dry run (see what would happen)
python3 backend/app/db/run_migrations.py --dry-run

# Run a specific migration
python3 backend/app/db/run_migrations.py --migration add_gender_column

# List all available migrations
python3 backend/app/db/run_migrations.py --list
```

#### LLM Provider Issues
```bash
# Check provider availability
python -c "from backend.app.services.llm_providers import LLMProviderFactory; print(LLMProviderFactory.get_available_providers())"

# Test specific provider
python -c "from backend.app.services.llm_providers import create_llm_provider; provider = create_llm_provider('ollama'); print('Available:', provider.is_available())"

# Check if legacy service exists (should return no results)
python -c "import os; print('Legacy llm_service.py exists:', os.path.exists('backend/app/services/llm_service.py'))"
```

#### HuggingFace Issues
```bash
# Check if transformers is installed
python -c "import transformers; print('Transformers available')"

# Check GPU availability for model acceleration
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test model loading
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM3-3B'); print('Model loaded successfully')"
```

#### Frontend Issues
```bash
cd test-frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Environment Variables

The project now uses a comprehensive FastAPI settings system with environment-specific configuration. Create `.env` file in project root:

```env
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
MINDORA_DATABASE_URL=postgresql://postgres:12345@localhost:5100/postgres

# LLM Providers
MINDORA_MODEL_OPENAI_API_KEY=your-openai-key
MINDORA_MODEL_GROQ_API_KEY=your-groq-key
MINDORA_MODEL_OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace Configuration (optional)
MINDORA_MODEL_DEFAULT_NAME=HuggingFaceTB/SmolLM3-3B  # Default: SmolLM3-3B

# Optional
MINDORA_MODEL_VLLM_BASE_URL=http://localhost:8001/v1
```

#### New FastAPI Settings System

The project uses a comprehensive FastAPI settings system using Pydantic BaseSettings. This provides:

- **Type Safety**: All configuration is strongly typed with validation
- **Environment Management**: Separate configurations for development, testing, and production
- **Secure Secrets Management**: API keys and sensitive data through environment variables
- **Hierarchical Configuration**: Organized settings categories (model, performance, safety, cultural, database)
- **IDE Support**: Autocompletion and documentation for all settings

##### Environment-Specific Configuration

The system automatically loads configuration based on the `ENVIRONMENT` variable:

- **Development**: `.env.development`
- **Testing**: `.env.testing`
- **Production**: `.env.production`

##### Configuration Categories

All settings use the `MINDORA_` prefix and are organized by category:

- **Model Settings** (`MINDORA_MODEL_*`): LLM configuration, providers, API keys
- **Performance Settings** (`MINDORA_PERFORMANCE_*`): Timeouts, limits, RAG configuration
- **Safety Settings** (`MINDORA_SAFETY_*`): Content filtering, crisis detection
- **Cultural Settings** (`MINDORA_CULTURAL_*`): Rwanda-specific context and resources
- **Database Settings** (`MINDORA_DATABASE_*`): Database connection and pool settings

For detailed documentation, see [`backend/app/settings/README.md`](backend/app/settings/README.md).

### Logs and Debugging

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Debug mode
PYTHONPATH=/path/to/mindora python -m debugpy --listen 5678 backend/app/main.py
```

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use ESLint for React/JavaScript code
- Add tests for new features
- Update documentation for API changes

### Adding New LLM Providers

1. Create provider class in `backend/app/services/llm_providers.py`
2. Implement `LLMProvider` interface
3. Add to `LLMProviderFactory.PROVIDERS`
4. Add tests in `tests/unit/test_llm_providers.py`

**Note**: The system uses `llm_service.py` as the main LLM orchestrator with gender-aware, culturally sensitive responses. All providers (Ollama, OpenAI, Groq, HuggingFace with SmolLM3-3B default) are supported through the factory pattern.

**HuggingFace Provider Tips**:
- Default model: SmolLM3-3B for optimal conversational AI and cultural sensitivity
- Use `transformers` pipeline for text generation with gender-aware responses
- Support both CPU and GPU inference (GPU recommended for better performance)
- Handle model loading and tokenization automatically
- Implement proper error handling for model availability
- Gender-aware cultural context with natural Kinyarwanda addressing

### Working with the New Settings System

The new FastAPI settings system provides a type-safe way to manage configuration:

#### Adding New Configuration Options

1. **Add to appropriate settings model** in `backend/app/settings/`:
    ```python
    # Example: Adding a new model setting
    class ModelSettings(BaseAppSettings):
        new_setting: str = Field(
            default="default_value",
            env="MINDORA_MODEL_NEW_SETTING",
            description="Description of the new setting"
        )
    ```

2. **Add to environment files**:
    - `.env.development`
    - `.env.testing`
    - `.env.production`

3. **Use in services**:
    ```python
    from backend.app.settings import settings

    def my_service():
        value = settings.model.new_setting
    ```

#### Configuration Best Practices

- Use the `MINDORA_` prefix for all environment variables
- Group related settings in the same model file
- Provide sensible defaults for all settings
- Add validation for critical settings
- Document all settings with descriptions
- Use environment-specific overrides when needed

For detailed documentation, see:
- [`backend/app/settings/README.md`](backend/app/settings/README.md) - Overview
- [`backend/app/settings/IMPLEMENTATION_GUIDE.md`](backend/app/settings/IMPLEMENTATION_GUIDE.md) - Implementation details

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rwanda Ministry of Health for mental health resources
- WHO for mhGAP guidelines
- LangChain community for LLM integration tools
- FastAPI community for the excellent web framework
- All contributors to open-source mental health initiatives
