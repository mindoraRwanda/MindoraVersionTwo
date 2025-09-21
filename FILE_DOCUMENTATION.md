# MINDORA - Detailed File Documentation

## Root Level Files

### `docker-compose.yml`
**Purpose**: Multi-service orchestration configuration
**Services Defined**:
- `postgres`: PostgreSQL database (port 5100)
- `qdrant`: Vector database for embeddings (ports 6333/6334)
- `vllm`: GPU-accelerated LLM service (port 8001)
- `backend`: FastAPI application (port 8000)
- `frontend`: React application (port 3000)

**Key Configurations**:
- Volume persistence for databases
- GPU resource allocation for vLLM
- Service dependencies and networking
- Environment variable injection

### `requirements.txt`
**Purpose**: Python backend dependencies specification
**Key Dependencies**:
- `fastapi==0.115.12`: Web framework
- `langchain==0.3.21`: LLM orchestration
- `qdrant-client==1.15.0`: Vector database client
- `sentence-transformers==5.0.0`: Text embeddings
- `transformers==4.50.3`: HuggingFace models
- `sqlalchemy==2.0.39`: Database ORM
- `psycopg2==2.9.10`: PostgreSQL adapter
- `bcrypt==4.0.1`: Password hashing
- `python-jose==3.5.0`: JWT handling
- `nemoguardrails==0.15.0`: AI safety guardrails

### `constraints.txt`
**Purpose**: Version constraints for dependency resolution
**Contents**: Specific versions for protobuf and OpenTelemetry components

### `package.json` (Root)
**Purpose**: Legacy React configuration (superseded by test-frontend/package.json)
**Dependencies**: Basic React setup with Tailwind CSS

---

## Backend Structure

### `backend/Dockerfile`
**Purpose**: Backend container configuration
**Base Image**: python:3.10-slim
**Features**:
- Build tools installation (gcc, g++, cmake)
- Python dependency installation
- Application code copying
- Port 8000 exposure
- Uvicorn server startup

### `backend/Scrap.py`
**Purpose**: Automated PDF document scraping utility
**Functionality**:
- Google search integration for mental health PDFs
- Automated download from WHO, APA, SAMHSA sources
- Target folder: Mental health knowledge base
- Search queries for screening tools and assessments

### `backend/app/main.py`
**Purpose**: FastAPI application entry point and lifecycle management
**Key Features**:
- **FastAPI Lifespan Management**: Initializes and shuts down services gracefully.
- **Service Initialization**:
    - `LLMService`: The core language model service.
    - `EmotionClassifier`: The emotion detection model.
    - `RAGService`: The retrieval-augmented generation service.
- **CORS Middleware**: Allows cross-origin requests from the frontend.
- **Router Registration**: Includes all API endpoints.
- **Database Initialization**: Creates database tables on startup.

**Routers Included**:
- Authentication router (`/auth`)
- Chat router (`/api`)
- Emotion analysis router
- Mental health insights router (`/insights`)

### `backend/app/__init__.py`
**Purpose**: Python package initialization (empty file)

---

## Authentication Module

### `backend/app/auth/auth_router.py`
**Purpose**: Core authentication and conversation management endpoints
**Endpoints**:
- `POST /auth/signup`: User registration with JWT token generation
- `POST /auth/login`: User authentication and token issuance
- `POST /auth/conversations`: Create new conversation
- `GET /auth/conversations`: List user conversations
- `GET /auth/conversations/{id}/messages`: Retrieve conversation messages
- `POST /auth/messages`: Send message with AI response generation
- `GET /auth/context`: Get recent conversation context
- `DELETE /auth/conversations/{id}`: Delete conversation and related data

**Key Features**:
- JWT-based authentication
- Optimized message pipeline with performance tracking
- Emotion detection integration
- Conversation history management
- Database transaction optimization

### `backend/app/auth/emotion_router.py`
**Purpose**: Mental health analysis and emotion detection endpoints
**Endpoints**:
- `POST /emotion`: Single emotion classification
- `POST /analyze`: Comprehensive mental health signal analysis
- `POST /detect/medications`: Medication mention extraction
- `POST /detect/suicide-risk`: Suicide risk assessment
- `POST /reindex`: Rebuild vector knowledge base (admin)

### `backend/app/auth/schemas.py`
**Purpose**: Pydantic models for request/response validation
**Models Defined**:
- `UserCreate`: Registration data validation
- `UserLogin`: Login credentials validation
- `TokenResponse`: JWT token response format
- `MessageCreate`: Message creation validation
- `MessageOut`: Message response format
- `ConversationOut`: Conversation response format
- `EmotionRequest`: Emotion analysis request
- `AnalysisRequest`: Comprehensive analysis request

**Validation Features**:
- Email format validation
- Username length constraints (3-20 characters)
- Password minimum length (6 characters)
- Message content validation

### `backend/app/auth/utils.py`
**Purpose**: Authentication utilities and security functions
**Functions**:
- `hash_password()`: bcrypt password hashing
- `verify_password()`: Password verification
- `create_access_token()`: JWT token generation
- `get_current_user()`: Token validation and user extraction

**Security Configuration**:
- HS256 algorithm for JWT
- 360-minute token expiration
- OAuth2 password bearer scheme
- Environment-based secret key

---

## Database Layer

### `backend/app/db/database.py`
**Purpose**: SQLAlchemy database configuration and connection management
**Configuration**:
- PostgreSQL connection string
- Session factory creation
- Declarative base definition
- Database session dependency injection

**Connection Details**:
- Host: localhost
- Port: 5432 (5100 in Docker)
- Database: postgres
- Credentials: postgres/12345

### `backend/app/db/models.py`
**Purpose**: SQLAlchemy ORM models for database schema
**Models Defined**:

#### User Model
- Primary key: `id`
- Fields: `username`, `email` (unique), `password` (hashed), `created_at`
- Relationships: One-to-many with conversations

#### Conversation Model
- Primary key: `id`
- Foreign key: `user_id`
- Fields: `started_at`, `last_activity_at`
- Relationships: Many-to-one with user, one-to-many with messages

#### Message Model
- Primary key: `id`
- Foreign key: `conversation_id`
- Fields: `sender` ("user"/"bot"), `content`, `timestamp`
- Relationships: Many-to-one with conversation

#### EmotionLog Model
- Primary key: `id`
- Foreign keys: `user_id`, `conversation_id`
- Fields: `input_text`, `detected_emotion`, `timestamp`
- Purpose: Track emotional states throughout conversations

---

## Core Services

### `backend/app/services/llm_service_refactored.py`
**Purpose**: Central LLM service, refactored for clarity and maintainability.
**Key Classes**:

#### LLMService
**Initialization**:
- Managed by the FastAPI `lifespan` event.
- `initialize()`: Checks for model availability and sets up safety guardrails.
- Supports both standard and vLLM backends.

**Core Methods**:
- `generate_response()`: The main entry point for generating a response.
- `_get_contextual_response_approach()`: Determines the best response strategy based on emotion and context.
- `_apply_safety_guardrails()`: Ensures the response is safe and appropriate.
- `_fetch_recent_conversation()`: Retrieves conversation history for context.

**Safety Features**:
- NeMo Guardrails integration
- Crisis intervention detection
- Inappropriate content filtering
- Medical advice boundaries
- Cultural sensitivity enforcement

**Performance Optimizations**:
- Conversation history limiting (15 messages)
- Fast path for simple messages
- Batch database operations
- Comprehensive timing metrics

### `backend/app/services/rag_service.py`
**Purpose**: Document processing and vector embedding generation
**Key Functions**:
- `initialize_rag_service()`: Initializes the service, processes PDFs, and loads them into Qdrant.
- `process_file()`: Processes a single PDF file.
- `process_embeddings_in_batches()`: Generates embeddings in batches to conserve memory.
- `create_collection_if_not_exists()`: Manages the Qdrant collection.

**Configuration**:
- PDF source folder: Configurable path
- Collection name: "therapy_knowledge_base"
- Batch size: 16 documents
- Max chunks per file: 1000
- Chunk size: 400 characters with 40-character overlap

**Features**:
- PyMuPDF for PDF text extraction
- RecursiveCharacterTextSplitter for intelligent chunking
- Sentence-transformers (all-MiniLM-L6-v2) for embeddings
- GPU memory management
- Progress tracking and error handling
- Batch upload to Qdrant

### `backend/app/services/retriever_service.py`
**Purpose**: High-performance vector similarity search
**Key Features**:
- Model caching across instances
- LRU cache for query encodings
- Optimized HNSW search parameters
- Sub-second search performance

**Configuration**:
- Default host: 127.0.0.1:6333
- Collection: "therapy_knowledge_base"
- Embedding model: all-MiniLM-L6-v2
- Search parameters: hnsw_ef=64, exact=False

### `backend/app/services/emotion_classifier.py`
**Purpose**: Real-time emotion classification from text
**Key Functions**:
- `initialize_emotion_classifier()`: Loads the model and pre-computes embeddings.
- `classify_emotion()`: Classifies the emotion of a given text.

**Emotion Categories** (10 total):
- sadness, anxiety, stress, fear, anger
- guilt, craving, numbness, joy, neutral

**Technical Implementation**:
- **Model**: `BAAI/bge-small-en` for sentence embeddings.
- **Method**: Cosine similarity against pre-computed emotion examples.
- **Performance**: GPU acceleration and model caching.

**Example Patterns**:
- Sadness: "I feel like crying all the time"
- Anxiety: "I'm always overthinking everything"
- Stress: "I feel like I'm drowning in responsibilities"

### `backend/app/services/chatbot_insights_pipeline.py`
**Purpose**: Comprehensive mental health signal detection
**Analysis Components**:
- Emotion classification (DistilRoBERTa)
- Sentiment analysis (DistilBERT)
- Toxicity detection (Toxic-BERT)
- Medication mention detection
- Suicide risk assessment

**Medication Keywords**:
- Common psychiatric medications (lithium, prozac, zoloft, xanax, etc.)
- Pattern matching in user text

**Suicide Risk Indicators**:
- Keywords: "kill myself", "suicide", "end it all", "no way out"
- Binary risk assessment (high/low)

**Dataset Integration**:
- Mental health FAQ datasets
- LLaMA2 mental health dialogues
- Curated psychology instructions
- Social media mental health data

---

## Router Layer

### `backend/app/routers/chat_router.py`
**Purpose**: Simple chat interface (legacy implementation)
**Endpoint**: `POST /api/chat`
**Features**:
- Basic conversation storage
- LLM service integration
- Conversation ID generation
- Message history tracking

**Note**: This is a simplified interface; main chat functionality is in auth_router.py

---

## Life Activities Module

### `backend/app/LifeActivities/fitlife_emotional_dataset.py`
**Purpose**: Fitness and emotional well-being analysis
**Key Functions**:
- `load_dataset()`: CSV data loading and preprocessing
- `top_mood_boosters()`: Activity ranking by mood improvement
- `top_stress_reducers()`: Activity ranking by stress reduction
- `recommend_by_stats()`: Personalized activity recommendations
- `train_and_save_model()`: ML model training for emotion prediction
- `predict_emotion()`: Activity-based emotion prediction

**ML Pipeline**:
- Features: activity_category, sub-category, intensity, mood_before, stress_level
- Target: primary_emotion
- Algorithm: Random Forest Classifier
- Model persistence with joblib

---

## Frontend Structure

### `test-frontend/Dockerfile`
**Purpose**: Frontend container configuration
**Base Image**: node:18
**Process**: npm install → copy source → expose port 3000 → npm start

### `test-frontend/package.json`
**Purpose**: Frontend dependency management
**Key Dependencies**:
- `react@18.3.1`: Core React library
- `react-dom@18.3.1`: DOM rendering
- `react-router-dom@6.30.1`: Client-side routing
- `axios@1.10.0`: HTTP client
- `tailwindcss@4.1.11`: Utility-first CSS framework

**Scripts**:
- `start`: Development server
- `build`: Production build
- `test`: Test runner

### `test-frontend/src/App.js`
**Purpose**: Main React application with routing configuration
**Routes Defined**:
- `/`: Login page
- `/register`: Registration page
- `/chats`: Chat list (unused)
- `/chat/:chatId`: Main chat interface

### `test-frontend/src/api/api.js`
**Purpose**: Centralized API client with authentication
**Functions**:
- `login()`: User authentication
- `register()`: User registration
- `getChats()`: Conversation list retrieval
- `startNewChat()`: New conversation creation
- `sendMessage()`: Message sending with response handling
- `getMessages()`: Conversation message retrieval
- `fetchChatContext()`: Recent conversation context

**Authentication**: Bearer token from localStorage

### `test-frontend/src/pages/Login.jsx`
**Purpose**: User authentication interface
**Features**:
- Email/password validation
- JWT token storage
- Automatic conversation routing
- Error handling with user feedback
- Responsive design with gradient styling

**Post-Login Flow**:
1. Fetch existing conversations
2. Navigate to latest conversation OR
3. Create new conversation if none exist

### `test-frontend/src/pages/Register.jsx`
**Purpose**: User registration interface
**Features**:
- Username, email, password validation
- Automatic first conversation creation
- Error handling with detailed feedback
- Consistent styling with login page

### `test-frontend/src/pages/ChatList.jsx`
**Purpose**: Conversation management interface (currently unused)
**Features**:
- Conversation listing
- New chat creation
- Chat deletion with confirmation
- Navigation to specific conversations

### `test-frontend/src/pages/ChatDashboard.jsx`
**Purpose**: Main chat interface - the core user experience
**Key Features**:

#### Sidebar
- Conversation list with preview text
- New chat creation button
- Logout functionality
- Delete conversation with confirmation
- Active conversation highlighting

#### Chat Panel
- Message display with sender differentiation
- Real-time typing indicators
- Emotion detection display
- Automatic scrolling to latest messages
- Message timestamp display

#### Input System
- Textarea with auto-resize
- Enter to send, Shift+Enter for new line
- Send button with loading states
- Message validation

**Styling**: Custom CSS with purple gradient theme, responsive design

**Performance Features**:
- Optimistic UI updates
- Efficient re-rendering
- Local state management
- Error boundary handling

---

## Utility and Configuration Files

### `test-frontend/src/index.js`
**Purpose**: React application entry point
**Functionality**: Root component rendering with StrictMode

### `test-frontend/src/index.css`
**Purpose**: Global styles and Tailwind CSS integration
**Contents**: Tailwind directives and custom global styles

### `test-frontend/public/`
**Purpose**: Static assets directory
**Contents**: 
- `index.html`: HTML template
- `favicon.ico`: Browser icon
- `manifest.json`: PWA configuration
- Logo images (192px, 512px)
- `robots.txt`: Search engine directives

---

## Vector Storage Directories

### `backend/app/services/mental_health_knowledge/`
**Purpose**: Qdrant vector database storage
**Contents**: Binary files for vector indices, metadata, and embeddings
**Structure**: UUID-named directories with:
- `data_level0.bin`: Vector data
- `header.bin`: Index headers
- `index_metadata.pickle`: Metadata
- `length.bin`: Vector lengths
- `link_lists.bin`: HNSW graph links

### `backend/mental_health_knowledge/`
**Purpose**: Additional vector storage location
**Contents**: Similar structure to services directory

---

## Development and Testing Files

### `test-frontend/src/App.test.js`
**Purpose**: React component testing
**Framework**: Jest with React Testing Library

### `test-frontend/src/setupTests.js`
**Purpose**: Test environment configuration
**Contents**: Jest DOM setup

### `test-frontend/src/reportWebVitals.js`
**Purpose**: Performance monitoring
**Functionality**: Web vitals reporting for optimization

---

## Configuration Files

### `test-frontend/src/postcss.config.js`
**Purpose**: PostCSS configuration for Tailwind CSS
**Plugins**: Tailwind CSS and Autoprefixer

### `test-frontend/src/tailwind.config.js`
**Purpose**: Tailwind CSS customization
**Configuration**: Content paths, theme extensions, plugins

### `.gitignore`
**Purpose**: Git ignore patterns
**Contents**: Node modules, build artifacts, environment files, IDE files

---

## Summary

This MINDORA project represents a sophisticated mental health chatbot system with:

- **434 lines** of comprehensive documentation
- **50+ files** across backend and frontend
- **Advanced AI integration** with safety guardrails
- **Cultural sensitivity** for Rwandan context
- **Production-ready architecture** with Docker deployment
- **Comprehensive security** with JWT authentication
- **Real-time emotion detection** and mental health analysis
- **Vector-based knowledge retrieval** from mental health documents
- **Responsive React frontend** with modern UX patterns

The system successfully combines cutting-edge AI technology with practical mental health support, creating a culturally-appropriate therapeutic companion for Rwandan youth.