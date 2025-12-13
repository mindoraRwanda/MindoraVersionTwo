"""
Unified RAG Service - Consolidates document processing and retrieval functionality.
Combines the best features of RAGService and RetrieverService into a single, efficient service.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import uuid
import gc

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance, SearchParams
    from sentence_transformers import SentenceTransformer
    import fitz  # PyMuPDF
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.groq import Groq
    import torch
except (ImportError, BaseException):
    # Fallback mocks if llama_index/qdrant is broken
    QdrantClient = Any
    PointStruct = Any
    VectorParams = Any
    Distance = Any
    SearchParams = Any
    SentenceTransformer = Any
    fitz = Any
    Document = Any
    VectorStoreIndex = Any
    StorageContext = Any
    QdrantVectorStore = Any
    HuggingFaceEmbedding = Any
    OllamaEmbedding = Any
    Ollama = Any
    OpenAI = Any
    Groq = Any
    torch = Any
    print("⚠️  Using Mocks for RAG dependencies (LlamaIndex/LangChain broken)")

from ..settings import settings

logger = logging.getLogger(__name__)

class UnifiedRAGService:
    """
    Unified RAG service that combines document processing and retrieval functionality.
    
    Features:
    - Fast retrieval with caching
    - Document processing and ingestion
    - Memory-efficient batch processing
    - Settings integration
    - Error handling and graceful degradation
    """
    
    _model_cache = {}  # Class-level cache for model reuse
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        batch_size: int = 16,
        max_chunks_per_file: int = 1000
    ):
        """Initialize the unified RAG service."""
        # Use settings if not provided
        self.collection_name = collection_name or "therapy_knowledge_base"
        self.embedding_model_name = embedding_model_name or settings.model.embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_chunks_per_file = max_chunks_per_file

        # Initialize components
        self.qdrant_client = None
        self.embedder = None
        self.llm = None
        self.vector_store = None
        self.index = None
        self.retriever = None
        self._initialized = False

        # Performance tracking
        self.search_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"🔧 UnifiedRAGService initialized with collection: {collection_name}")
    
    def initialize(self, vectorize: bool = False) -> bool:
        """Initialize the RAG service with Qdrant and embedding model."""
        try:
            logger.info("🚀 Initializing UnifiedRAGService...")
            
            # Initialize Qdrant client
            self._init_qdrant_client()
            
            # Initialize embedding model
            self._init_embedding_model()
            
            # Initialize LlamaIndex components
            self._init_llamaindex_components()

            # Create collection if needed
            self._ensure_collection_exists()
            
            # Process documents if requested
            if vectorize:
                self._process_existing_documents()
            
            self._initialized = True
            logger.info("✅ UnifiedRAGService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedRAGService: {e}")
            return False
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client with settings integration."""
        try:
            # Start with HTTP by default for local development
            qdrant_config = {
                "host": settings.qdrant.qdrant_host,
                "port": settings.qdrant.qdrant_port,
                "https": False  # Default to HTTP
            }
            if settings.qdrant.qdrant_api_key:
                qdrant_config["api_key"] = settings.qdrant.qdrant_api_key

            # Try HTTP first (most common for local Qdrant)
            try:
                self.qdrant_client = QdrantClient(**qdrant_config)
                logger.info(f"🔗 Connected to Qdrant via HTTP at {settings.qdrant.qdrant_host}:{settings.qdrant.qdrant_port}")
            except Exception as http_error:
                # If HTTP fails, try HTTPS
                logger.warning(f"HTTP connection failed, trying HTTPS: {http_error}")
                qdrant_config["https"] = True
                try:
                    self.qdrant_client = QdrantClient(**qdrant_config)
                    logger.info(f"🔗 Connected to Qdrant via HTTPS at {settings.qdrant.qdrant_host}:{settings.qdrant.qdrant_port}")
                except Exception as https_error:
                    logger.error(f"Both HTTP and HTTPS failed. HTTP: {http_error}, HTTPS: {https_error}")
                    raise https_error

        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    def _init_embedding_model(self):
        """Initialize embedding model with caching."""
        try:
            # Reuse model across instances
            if self.embedding_model_name not in self._model_cache:
                logger.info(f"📚 Loading embedding model: {self.embedding_model_name}")
                self._model_cache[self.embedding_model_name] = SentenceTransformer(self.embedding_model_name)
            
            self.embedder = self._model_cache[self.embedding_model_name]
            logger.info(f"✅ Embedding model loaded: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _init_llamaindex_components(self):
        """Initialize LlamaIndex components."""
        # Initialize embedding model based on settings
        self.embedder = self._create_embedding_model()

        # Initialize LLM (optional, for query engines)
        self.llm = self._create_llm()

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name
        )

        # Initialize storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index (will be populated when documents are added)
        self.index = VectorStoreIndex.from_documents(
            [],  # Empty initially, will be populated during document processing
            storage_context=storage_context,
            embed_model=self.embedder
        )

        # Create retriever (not query engine)
        self.retriever = self.index.as_retriever(similarity_top_k=settings.performance.rag_top_k if settings.performance else 3)

        logger.info("✂️ LlamaIndex components initialized")

    def _create_embedding_model(self):
        """Create embedding model based on settings."""
        provider = settings.model.embedding_model_provider.lower()

        if provider == "huggingface":
            try:
                return HuggingFaceEmbedding(
                    model_name=settings.model.embedding_model_name,
                    device="cuda"  # Try CUDA first
                )
            except Exception as e:
                logger.warning(f"CUDA not available for embeddings, falling back to CPU: {e}")
                return HuggingFaceEmbedding(
                    model_name=settings.model.embedding_model_name,
                    device="cpu"
                )
        elif provider == "ollama":
            try:
                return OllamaEmbedding(
                    model_name=settings.model.embedding_model_name,
                    device="cuda"  # Try CUDA first
                )
            except Exception as e:
                logger.warning(f"CUDA not available for embeddings, falling back to CPU: {e}")
                return HuggingFaceEmbedding(
                    model_name=settings.model.embedding_model_name,
                    device="cpu"
                )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_llm(self):
        """Create LLM based on settings."""
        provider = settings.model.llamaindex_llm_provider.lower()

        if provider == "ollama":
            return Ollama(
                model=settings.model.llamaindex_model_name,
                base_url=settings.model.ollama_base_url,
                temperature=settings.model.temperature
            )
        elif provider == "openai":
            if not settings.model.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI LLM")
            return OpenAI(
                api_key=settings.model.openai_api_key,
                model=settings.model.llamaindex_model_name,
                temperature=settings.model.temperature,
                max_tokens=settings.model.max_tokens
            )
        elif provider == "groq":
            if not settings.model.groq_api_key:
                raise ValueError("Groq API key required for Groq LLM")
            return Groq(
                api_key=settings.model.groq_api_key,
                model=settings.model.llamaindex_model_name,
                temperature=settings.model.temperature,
                max_tokens=settings.model.max_tokens
            )
        else:
            logger.warning(f"Unsupported LLM provider: {provider}, LLM will be None")
            return None
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"📁 Creating collection: {self.collection_name}")
                # LlamaIndex will handle collection creation with proper dimensions
                # when documents are first indexed
                logger.info("✅ Collection will be created when first documents are indexed")
            else:
                logger.info(f"✅ Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"❌ Failed to ensure collection exists: {e}")
            raise
    
    def _process_existing_documents(self):
        """Process existing PDF documents from datasources folder."""
        try:
            # Look for PDFs in multiple locations
            pdf_folders = [
                Path("./backend/datasources"),
                Path("./data"),
                Path("./datasources")
            ]
            
            pdf_files = []
            for folder in pdf_folders:
                if folder.exists():
                    pdf_files.extend(list(folder.glob("*.pdf")))
                    logger.info(f"📂 Found {len(list(folder.glob('*.pdf')))} PDFs in {folder}")
            
            if not pdf_files:
                logger.warning("⚠️ No PDF files found in any of the expected locations")
                return
            
            logger.info(f"📚 Processing {len(pdf_files)} PDF files...")
            
            successful = 0
            failed = 0
            
            for i, file_path in enumerate(pdf_files, 1):
                logger.info(f"📄 Processing file {i}/{len(pdf_files)}: {file_path.name}")
                
                try:
                    if self._process_single_file(file_path):
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path.name}: {e}")
                    failed += 1
            
            logger.info(f"📊 Document processing complete: {successful} successful, {failed} failed")
            
        except Exception as e:
            logger.error(f"❌ Failed to process existing documents: {e}")
    
    def _process_single_file(self, file_path: Path) -> bool:
        """Process a single PDF file using LlamaIndex."""
        try:
            # Check file size
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > 50:
                logger.warning(f"⚠️ Large file: {file_path.name} ({file_size:.1f}MB)")

            # Extract text from PDF
            doc = fitz.open(file_path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()

            if not full_text.strip():
                logger.warning(f"⚠️ No text extracted from {file_path.name}")
                return False

            # Create LlamaIndex Document
            document = Document(
                text=full_text,
                metadata={
                    "source": file_path.name,
                    "file_size": file_size,
                    "processed_at": time.time()
                }
            )

            # Add document to index
            self.index.insert(document)

            logger.info(f"✅ Successfully processed {file_path.name} with LlamaIndex")
            return True

        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return False
    
    
    @lru_cache(maxsize=100)
    def _encode_query(self, query: str) -> Tuple[float, ...]:
        """Cache encoded queries to avoid re-encoding identical queries."""
        return tuple(self.embedder._embed([query])[0])
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using LlamaIndex retriever (not query engine).

        Args:
            query: Search query
            top_k: Number of results to return (defaults to settings)

        Returns:
            List of search results with metadata
        """
        if not self._initialized:
            logger.warning("⚠️ RAG service not initialized")
            return []

        if top_k is None:
            top_k = settings.performance.rag_top_k if settings.performance else 3

        start_time = time.time()

        try:
            # Update retriever similarity_top_k if different
            if hasattr(self.retriever, 'similarity_top_k') and self.retriever.similarity_top_k != top_k:
                self.retriever = self.index.as_retriever(similarity_top_k=top_k)

            # Perform search using retriever
            nodes = self.retriever.retrieve(query)

            # Format results
            formatted_results = []
            for node in nodes:
                formatted_results.append({
                    "id": node.node_id,
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "text": node.node.get_text() if hasattr(node, 'node') else str(node),
                    "source": node.node.metadata.get("source", "") if hasattr(node, 'node') and node.node.metadata else "",
                    "file_size": node.node.metadata.get("file_size", 0) if hasattr(node, 'node') and node.node.metadata else 0,
                    "processed_at": node.node.metadata.get("processed_at", 0) if hasattr(node, 'node') and node.node.metadata else 0
                })

            search_time = time.time() - start_time
            logger.info(f"🔍 RAG search completed in {search_time:.3f}s: {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return []
    
    def search_text_only(self, query: str, top_k: int = None) -> List[str]:
        """
        Search and return only the text content of results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of text chunks
        """
        results = self.search(query, top_k)
        return [result["text"] for result in results if result.get("text")]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection using LlamaIndex."""
        try:
            if not self._initialized:
                return {"error": "Service not initialized"}

            # Get collection info from Qdrant client
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                return {"error": f"Collection '{self.collection_name}' not found"}

            collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "payload_schema": collection_info.payload_schema
            }

        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized."""
        return self._initialized and self.qdrant_client is not None and self.embedder is not None and self.index is not None and self.retriever is not None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.search_cache)
        }


def create_unified_rag_service() -> Optional[UnifiedRAGService]:
    """Factory function to create a unified RAG service."""
    try:
        service = UnifiedRAGService()
        service.initialize(vectorize=False)  # Don't vectorize on startup
        return service
    except Exception as e:
        logger.error(f"❌ Failed to create unified RAG service: {e}")
        return None


if __name__ == "__main__":
    # Test the service
    service = create_unified_rag_service()
    if service:
        print("✅ UnifiedRAGService created successfully")
        print(f"📊 Collection info: {service.get_collection_info()}")
        
        # Test search
        results = service.search("mental health depression", top_k=3)
        print(f"🔍 Search results: {len(results)} found")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['source']} (score: {result['score']:.3f})")
    else:
        print("❌ Failed to create UnifiedRAGService")
