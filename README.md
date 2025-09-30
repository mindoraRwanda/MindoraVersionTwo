# Mindora Conversational Therapy Chatbot

A comprehensive mental health support platform designed for Rwandan youth, featuring advanced AI capabilities, gender-aware cultural sensitivity, and natural conversational responses. Built with modern web technologies and multiple AI integrations for scalable, reliable mental health support that feels like talking to a culturally aware Rwandan brother or sister.

---

## Features

- **Multi-Provider AI Support**: Integration with Ollama, OpenAI, Groq, and HuggingFace (default: SmolLM3-3B) for flexible LLM deployment
- **Gender-Aware Cultural Context**: Personalized responses based on user gender with natural Kinyarwanda addressing ("murumuna" for brothers, "murumuna wanjye" for sisters)
- **Natural Conversational Responses**: AI speaks like a culturally aware Rwandan brother/sister rather than a clinical assistant
- **Advanced RAG System**: ChromaDB/Qdrant vector database with intelligent document retrieval
- **Emotion Detection**: Real-time emotion classification with gender-aware response adaptation
- **Safety & Ethics**: Comprehensive guardrails, crisis detection, and content filtering
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
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
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

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Run database migrations
python -c "from backend.app.db.database import engine; from backend.app.db import models; models.Base.metadata.create_all(bind=engine)"

# Start backend
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

#### Chat
- **POST** `/chat` - Send message and get AI response
- **GET** `/chat/history/{conversation_id}` - Get conversation history

#### Authentication
- **POST** `/auth/register` - User registration
- **POST** `/auth/login` - User login
- **POST** `/auth/logout` - User logout

#### Conversations
- **GET** `/conversations` - List user conversations
- **POST** `/conversations` - Create new conversation
- **DELETE** `/conversations/{id}` - Delete conversation

#### Messages
- **GET** `/messages/{conversation_id}` - Get messages in conversation
- **POST** `/messages` - Send message

#### Mental Health Insights
- **GET** `/insights/emotion` - Get emotion analysis
- **GET** `/insights/crisis-resources` - Get Rwanda crisis resources
- **GET** `/insights/cultural-context` - Get cultural context information

### Example Request

```bash
curl -X POST "http://localhost:8000/chat" \
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

The backend follows a modular architecture with clear separation of concerns:

```
backend/
├── __init__.py                    # Backend package initialization
├── main.py                        # FastAPI application entry point
├── Dockerfile                     # Container configuration
├── Scrap.py                       # Web scraping utilities
├── test_*.py                      # Standalone test files
├── .benchmarks/                   # Performance benchmarking data
├── .pytest_cache/                 # Pytest cache directory
├── app/                           # Main application package
│   ├── __init__.py               # App package initialization
│   ├── main.py                   # FastAPI app instance & routes
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
│   │   ├── chat_router.py        # Chat functionality
│   │   ├── conversations_router.py
│   │   └── messages_router.py
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── chatbot_insights_pipeline.py  # Crisis detection
│   │   ├── emotion_classifier.py         # Emotion analysis
│   │   ├── langgraph_state.py            # LangGraph state management
│   │   ├── llm_config.py                 # LLM configuration
│   │   ├── llm_cultural_context.py       # Cultural context handling
│   │   ├── llm_database_operations.py    # DB operations
│   │   ├── llm_model_manager.py          # Model management
│   │   ├── llm_providers.py              # Multi-provider support
│   │   ├── llm_safety.py                 # Safety guardrails
│   │   ├── llm_service_refactored.py     # Main LLM orchestrator
│   │   ├── query_validator_langgraph.py  # LangGraph validation
│   │   ├── query_validator.py            # Query validation
│   │   ├── rag_service.py                # RAG implementation
│   │   ├── retriever_service.py          # Document retrieval
│   │   └── mental_health_knowledge/      # Vector DB storage
│   └── utils/                    # Utility functions
│       └── __init__.py
├── requirements.txt              # Python dependencies
└── mental_health_knowledge/      # Vector database files
```

### Component Descriptions

#### **Core Application** (`app/`)
- **`main.py`**: FastAPI application instance with lifespan management, CORS configuration, and route registration
- **`auth/`**: Complete authentication system with JWT tokens, password hashing, and user management
- **`db/`**: Database configuration, connection management, and SQLAlchemy models for users, conversations, and messages
- **`models/`**: Additional data models and Pydantic schemas for API contracts

#### **AI Services** (`services/`)
- **`llm_service.py`**: Main orchestrator for LLM operations with full pipeline processing (renamed from refactored version)
- **`llm_providers.py`**: Factory pattern for multiple LLM providers (Ollama, OpenAI, Groq, vLLM)
- **`rag_service.py`**: Retrieval-Augmented Generation with vector similarity search
- **`emotion_classifier.py`**: Real-time emotion detection and classification
- **`query_validator*.py`**: Input validation and safety checking with LangGraph integration
- **`llm_safety.py`**: Content filtering and safety guardrails
- **`llm_cultural_context.py`**: Gender-aware Rwandan cultural context with natural Kinyarwanda addressing

#### **Prompt Management** (`prompts/`)
- **`system_prompts.py`**: Core system prompts with cultural context
- **`safety_prompts.py`**: Safety and ethical response templates
- **`cultural_context_prompts.py`**: Rwanda-specific cultural prompts
- **`response_approach_prompts.py`**: Emotion-based response strategies

#### **API Layer** (`routers/`)
- **`chat_router.py`**: Main chat functionality and message processing
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
- **Configuration**: Flexible configuration system with environment variables and external files
- **Cultural Context**: Gender-aware Rwandan cultural integration with natural Kinyarwanda addressing

### ⚠️ Legacy Code (Safe to Remove)
- **`test_*.py` in backend root**: Functionality moved to `/tests/` directory
- **`backend/.env`**: Environment config should be in project root
- **`backend/datasources/`**: Unused directory (use `backend/mental_health_knowledge/` instead)

### 🔧 Recent Updates
- **Gender-Aware Cultural Context**: Added personalized responses based on user gender with Kinyarwanda addressing
- **Natural Conversational Responses**: Enhanced AI to speak like a culturally aware Rwandan brother/sister
- **HuggingFace Default Provider**: Set SmolLM3-3B as default model for optimal privacy and cultural sensitivity
- **Configuration System Refactoring**: Replaced hardcoded values with flexible, structured configuration management
- **Enhanced Emotion Detection**: Integrated emotion analysis into LangGraph workflow for better response personalization
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

Create `.env` file in project root:

```env
# Database
DATABASE_URL=postgresql://postgres:12345@localhost:5100/postgres

# LLM Providers
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace Configuration (optional)
HUGGINGFACE_MODEL_PATH=HuggingFaceTB/SmolLM3-3B  # Default: SmolLM3-3B
HUGGINGFACE_DEVICE=cuda  # or "cpu"

# Optional
VLLM_BASE_URL=http://localhost:8001/v1
```

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

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rwanda Ministry of Health for mental health resources
- WHO for mhGAP guidelines
- LangChain community for LLM integration tools
- FastAPI community for the excellent web framework
- All contributors to open-source mental health initiatives
 


