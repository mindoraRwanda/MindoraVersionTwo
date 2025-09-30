from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, SearchRequest, PointStruct, SearchParams
from typing import List
import time
from functools import lru_cache
import hashlib

class RetrieverService:
    _model_cache = {}  # Class-level cache for model reuse
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        collection_name: str = "therapy_knowledge_base",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Reuse model across instances
        if embedding_model_name not in self._model_cache:
            print(f"Loading embedding model: {embedding_model_name}")
            self._model_cache[embedding_model_name] = SentenceTransformer(embedding_model_name)
        self.model = self._model_cache[embedding_model_name]

    @lru_cache(maxsize=100)
    def _encode_query(self, query: str):
        """Cache encoded queries to avoid re-encoding identical queries"""
        return tuple(self.model.encode(query).tolist())
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search Qdrant for chunks similar to the user query.
        Returns a list of relevant text chunks.
        """
        start_time = time.time()

        try:
            # Check if collection exists first
            collections = self.qdrant.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                print(f"❌ RAG: Collection '{self.collection_name}' not found. Available: {collection_names}")
                return []

            # Use cached encoding
            query_vector = list(self._encode_query(query))

            # Optimized search parameters for speed
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(
                    hnsw_ef=64,  # Reduced from 128 for faster search
                    exact=False  # Use approximate search for speed
                )
            )

            search_time = time.time() - start_time
            print(f"RAG search completed in {search_time:.3f}s")

            return [hit.payload["text"] for hit in results if hit.payload and "text" in hit.payload]

        except Exception as e:
            print(f"❌ RAG Error: {e}")
            return []
