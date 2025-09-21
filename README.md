# Mindora Conversational Therapy Chatbot

A mental health chatbot designed to support Rwandan youth using localized, evidence-based content. It uses Retrieval-Augmented Generation (RAG) with Qdrant, LangChain, and a locally hosted LLM (Ollama).

---

## Features

- Context-aware chatbot using real Rwandan mental health documents  
- Integrated with Qdrant vector DB for smart document retrieval  
- Local LLM using Ollama (LLaMA 3)  
- FastAPI backend + Node.js frontend  
- Automatically scrapes and indexes PDF documents  

---

## Prerequisites

- Python 3.10+  
- Node.js v18+  
- Docker Desktop  
- Ollama (local LLM)  
- Git  

---

## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/mindora-chatbot.git
cd mindora-chatbot
```

### Step 2: Backend Setup

```bash
cd backend
conda create -n mindora python=3.10
conda activate mindora
pip install -r requirements.txt
pip install fitz PyMuPDF sentence-transformers langchain-community qdrant-client
```

### Step 3: Run Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Step 4: Embed PDFs

Place your PDFs in:

```
C:/Users/STUDENT/Documents/MentalHealthChatbot 
```
Or if you placed them in a different file make sure you update this path
Then run:

```bash
cd backend/app/services
python rag_service.py
```

### Step 5: Start FastAPI Backend

```bash
cd backend
uvicorn app.main:app --reload
```

FastAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 6: Start Frontend

```bash
cd test-frontend
npm install
npm run dev
```

Frontend URL: [http://localhost:3000](http://localhost:3000)

---

## Test

**Endpoint:** `/api/chat`  
**Method:** POST

**Example Request Body:**

```json
{
  "message": "What does Rwanda’s mental health policy say about community-based care?",
  "conversation_id": "test1"
}
```

**Expected Response:**  
Returns context-rich response pulled from uploaded documents.

---

## Data Sources

**Total Collected:** 140

**Sources Include:**

- Rwanda Ministry of Health  
- Rwanda Biomedical Center  
- WHO mhGAP and Mental Health Atlas  
- APA and SAMHSA publications  
- PTSD, PHQ-9, GAD-7 screening tools  
- Academic articles  

---

## Adding More Documents

1. Run:

```bash
python backend/app/services/rag_service.py
```

 


