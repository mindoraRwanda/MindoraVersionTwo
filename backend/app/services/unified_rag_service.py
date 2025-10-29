# backend/app/services/unified_rag_service.py
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

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, SearchParams
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import torch

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
        collection_name: str = "therapy_knowledge_base",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        batch_size: int = 16,
        max_chunks_per_file: int = 1000
    ):
        """Initialize the unified RAG service."""
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_chunks_per_file = max_chunks_per_file
        
        # Initialize components
        self.qdrant_client = None
        self.embedder = None
        self.text_splitter = None
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
            
            # Initialize text splitter
            self._init_text_splitter()
            
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
    
    def _init_text_splitter(self):
        """Initialize text splitter for document processing."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"✂️ Text splitter initialized: {self.chunk_size} chars, {self.chunk_overlap} overlap")
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"📁 Creating collection: {self.collection_name}")
                embedding_dimension = self.embedder.get_sentence_embedding_dimension()
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
                )
                logger.info("✅ Collection created successfully")
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
        """Process a single PDF file."""
        try:
            # Check file size
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > 50:
                logger.warning(f"⚠️ Large file: {file_path.name} ({file_size:.1f}MB)")
            
            # Extract text from PDF
            doc = fitz.open(file_path)
            full_text = "\n".join([page.get_text() for page in doc if hasattr(page, 'get_text')])
            doc.close()
            
            if not full_text.strip():
                logger.warning(f"⚠️ No text extracted from {file_path.name}")
                return False
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            logger.info(f"✂️ Split into {len(chunks)} chunks")
            
            # Limit chunks for very large documents
            if len(chunks) > self.max_chunks_per_file:
                logger.warning(f"⚠️ Limiting chunks from {len(chunks)} to {self.max_chunks_per_file}")
                chunks = chunks[:self.max_chunks_per_file]
            
            # Generate embeddings in batches
            embeddings = self._generate_embeddings_batch(chunks)
            
            # Create and upload points
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path.name}_{i}"))
                points.append(PointStruct(
                    id=unique_id,
                    vector=vector,
                    payload={
                        "source": file_path.name,
                        "chunk_id": i,
                        "text": chunk,
                        "file_size": file_size,
                        "processed_at": time.time()
                    }
                ))
            
            # Upload in batches
            self._upload_points_batch(points)
            
            logger.info(f"✅ Successfully processed {file_path.name}: {len(points)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return False
    
    def _generate_embeddings_batch(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks in batches."""
        all_embeddings = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            logger.info(f"🔄 Processing embedding batch {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1}")
            
            try:
                batch_embeddings = self.embedder.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect()
                time.sleep(0.1)  # Prevent overheating
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {i//self.batch_size + 1}: {e}")
                if self.batch_size > 4:
                    logger.info("🔄 Retrying with smaller batch size...")
                    return self._generate_embeddings_batch(chunks, self.batch_size // 2)
                else:
                    raise e
        
        return all_embeddings
    
    def _upload_points_batch(self, points: List[PointStruct]):
        """Upload points to Qdrant in batches."""
        upload_batch_size = 100
        for i in range(0, len(points), upload_batch_size):
            batch_points = points[i:i + upload_batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )
            logger.info(f"📤 Uploaded batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")
    
    @lru_cache(maxsize=100)
    def _encode_query(self, query: str) -> Tuple[float, ...]:
        """Cache encoded queries to avoid re-encoding identical queries."""
        return tuple(self.embedder.encode(query).tolist())
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database.
        
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
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"❌ Collection '{self.collection_name}' not found")
                return []
            
            # Use cached encoding
            query_vector = list(self._encode_query(query))
            
            # Search with optimized parameters
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(
                    hnsw_ef=64,  # Optimized for speed
                    exact=False  # Use approximate search
                )
            )
            
            # Format results
            formatted_results = []
            for point in results:
                payload = getattr(point, 'payload', {})
                formatted_results.append({
                    "id": getattr(point, 'id', ''),
                    "score": getattr(point, 'score', 0.0),
                    "text": payload.get("text", "") if payload else "",
                    "source": payload.get("source", "") if payload else "",
                    "chunk_id": payload.get("chunk_id", 0) if payload else 0,
                    "file_size": payload.get("file_size", 0) if payload else 0,
                    "processed_at": payload.get("processed_at", 0) if payload else 0
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
        """Get information about the current collection."""
        try:
            if not self._initialized:
                return {"error": "Service not initialized"}
            
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
        return self._initialized and self.qdrant_client is not None and self.embedder is not None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.search_cache)
        }


def create_unified_rag_service() -> UnifiedRAGService:
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
