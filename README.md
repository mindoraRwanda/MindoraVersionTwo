# Mindora Conversational Therapy Chatbot

A comprehensive mental health support platform designed for Rwandan youth, featuring advanced AI capabilities, cultural sensitivity, and evidence-based therapeutic approaches. Built with modern web technologies and multiple AI integrations for scalable, reliable mental health support.

---

## Features

- **Multi-Provider AI Support**: Integration with Ollama, OpenAI, and Groq for flexible LLM deployment
- **Advanced RAG System**: ChromaDB/Qdrant vector database with intelligent document retrieval
- **Cultural Context Awareness**: Rwanda-specific cultural understanding and Ubuntu philosophy integration
- **Emotion Detection**: Real-time emotion classification and adaptive response generation
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

By default, the system uses Groq. To use other providers:

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

### Document Ingestion

```bash
# Place your mental health documents in a directory
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

```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── routers/                # API route handlers
│   │   ├── auth_router.py      # Authentication endpoints
│   │   ├── chat_router.py      # Chat functionality
│   │   ├── conversations_router.py
│   │   └── messages_router.py
│   ├── services/               # Business logic
│   │   ├── llm_service_refactored.py  # Main LLM orchestrator
│   │   ├── llm_providers.py    # Multi-provider LLM support
│   │   ├── rag_service.py      # Document retrieval
│   │   ├── emotion_classifier.py
│   │   ├── query_validator.py  # LangGraph integration
│   │   └── safety_manager.py   # Content filtering
│   ├── prompts/                # Prompt templates
│   ├── db/                     # Database models and config
│   └── auth/                   # Authentication logic
├── requirements.txt
└── Dockerfile
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

#### 4. vLLM (GPU Acceleration)
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

1. Place PDF documents in `backend/mental_health_knowledge/` directory
2. Run the document ingestion service:

```bash
python -m backend.app.services.rag_service
```
3. Documents will be automatically:
   - Parsed and chunked
   - Embedded using sentence transformers
   - Stored in vector database
   - Made available for RAG queries

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

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rwanda Ministry of Health for mental health resources
- WHO for mhGAP guidelines
- LangChain community for LLM integration tools
- FastAPI community for the excellent web framework
- All contributors to open-source mental health initiatives
 


