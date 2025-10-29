"""
Vector Database Population Utility

This module provides utilities to populate the vector database with documents
for the RAG (Retrieval-Augmented Generation) system.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.unified_rag_service import UnifiedRAGService, create_unified_rag_service
from app.settings.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorDBPopulator:
    """
    Utility class for populating the vector database with documents.
    
    This class handles the process of:
    1. Initializing the RAG service
    2. Processing documents from specified directories
    3. Generating embeddings and storing them in Qdrant
    4. Providing progress feedback and error handling
    """
    
    def __init__(
        self,
        collection_name: str = "therapy_knowledge_base",
        data_directories: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the Vector DB Populator.
        
        Args:
            collection_name: Name of the Qdrant collection
            data_directories: List of directories to scan for documents
            file_extensions: List of file extensions to process (default: ['.pdf'])
        """
        self.collection_name = collection_name
        self.data_directories = data_directories or [
            "./backend/datasources",
            "./data", 
            "./datasources",
            "./documents"
        ]
        self.file_extensions = file_extensions or ['.pdf', '.txt', '.md', '.docx']
        self.rag_service: Optional[UnifiedRAGService] = None
        
        logger.info(f"🔧 VectorDBPopulator initialized")
        logger.info(f"   Collection: {self.collection_name}")
        logger.info(f"   Data directories: {self.data_directories}")
        logger.info(f"   File extensions: {self.file_extensions}")
    
    def initialize_rag_service(self) -> bool:
        """
        Initialize the RAG service for vector database operations.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing RAG service...")
            
            self.rag_service = UnifiedRAGService(collection_name=self.collection_name)
            success = self.rag_service.initialize(vectorize=False)
            
            if success:
                logger.info("✅ RAG service initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize RAG service")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing RAG service: {e}")
            return False
    
    def find_documents(self) -> List[Path]:
        """
        Find all documents in the specified directories.
        
        Returns:
            List[Path]: List of document file paths
        """
        documents = []
        
        for directory in self.data_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"⚠️ Directory not found: {directory}")
                continue
            
            logger.info(f"📂 Scanning directory: {directory}")
            
            for ext in self.file_extensions:
                files = list(dir_path.glob(f"**/*{ext}"))
                documents.extend(files)
                logger.info(f"   Found {len(files)} {ext} files")
        
        logger.info(f"📊 Total documents found: {len(documents)}")
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get current collection statistics.
        
        Returns:
            Dict containing collection information
        """
        if not self.rag_service:
            return {"error": "RAG service not initialized"}
        
        return self.rag_service.get_collection_info()
    
    def populate_database(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Populate the vector database with documents.
        
        Args:
            force_reprocess: If True, reprocess all documents even if collection exists
            
        Returns:
            Dict containing processing results and statistics
        """
        if not self.rag_service:
            logger.error("❌ RAG service not initialized. Call initialize_rag_service() first.")
            return {"success": False, "error": "RAG service not initialized"}
        
        start_time = datetime.now()
        logger.info("🔄 Starting vector database population...")
        
        try:
            # Get initial stats
            initial_stats = self.get_collection_stats()
            initial_count = initial_stats.get('points_count', 0)
            
            if initial_count > 0 and not force_reprocess:
                logger.info(f"📊 Collection already contains {initial_count} vectors")
                logger.info("   Use force_reprocess=True to reprocess all documents")
                return {
                    "success": True,
                    "message": "Collection already populated",
                    "initial_count": initial_count,
                    "processing_time": 0
                }
            
            # Find documents
            documents = self.find_documents()
            
            if not documents:
                logger.warning("⚠️ No documents found to process")
                return {
                    "success": False,
                    "error": "No documents found",
                    "documents_found": 0
                }
            
            # Process documents by reinitializing with vectorize=True
            logger.info("📚 Processing documents and generating embeddings...")
            success = self.rag_service.initialize(vectorize=True)
            
            if not success:
                logger.error("❌ Failed to process documents")
                return {
                    "success": False,
                    "error": "Document processing failed"
                }
            
            # Get final stats
            final_stats = self.get_collection_stats()
            final_count = final_stats.get('points_count', 0)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("✅ Vector database population completed!")
            logger.info(f"   📊 Vectors added: {final_count - initial_count}")
            logger.info(f"   📊 Total vectors: {final_count}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
            
            return {
                "success": True,
                "documents_found": len(documents),
                "initial_count": initial_count,
                "final_count": final_count,
                "vectors_added": final_count - initial_count,
                "processing_time": processing_time,
                "collection_stats": final_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error during database population: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def test_search(self, query: str = "mental health support", top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Test the search functionality after population.
        
        Args:
            query: Test query string
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.rag_service:
            logger.error("❌ RAG service not initialized")
            return []
        
        logger.info(f"🔍 Testing search with query: '{query}'")
        results = self.rag_service.search(query, top_k=top_k)
        
        logger.info(f"📊 Search returned {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"   {i}. Score: {result['score']:.3f} | Source: {result['source']}")
        
        return results


def populate_vector_db(
    collection_name: str = "therapy_knowledge_base",
    data_directories: Optional[List[str]] = None,
    force_reprocess: bool = False,
    test_search: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to populate the vector database.
    
    Args:
        collection_name: Name of the Qdrant collection
        data_directories: List of directories to scan for documents
        force_reprocess: If True, reprocess all documents
        test_search: If True, run a test search after population
        
    Returns:
        Dict containing processing results
    """
    # Set environment if not already set
    if not os.getenv("ENVIRONMENT"):
        os.environ["ENVIRONMENT"] = "development"
    
    logger.info("🚀 Starting vector database population utility...")
    
    # Initialize populator
    populator = VectorDBPopulator(
        collection_name=collection_name,
        data_directories=data_directories
    )
    
    # Initialize RAG service
    if not populator.initialize_rag_service():
        return {"success": False, "error": "Failed to initialize RAG service"}
    
    # Populate database
    results = populator.populate_database(force_reprocess=force_reprocess)
    
    # Test search if requested and population was successful
    if test_search and results.get("success"):
        logger.info("🔍 Running test search...")
        search_results = populator.test_search()
        results["test_search_results"] = len(search_results)
    
    return results


if __name__ == "__main__":
    """
    Command-line interface for the vector database populator.
    
    Usage:
        python -m app.utils.vector_db_populator
        python -m app.utils.vector_db_populator --force
        python -m app.utils.vector_db_populator --collection my_collection
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate vector database with documents")
    parser.add_argument(
        "--collection", 
        default="therapy_knowledge_base",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing of all documents"
    )
    parser.add_argument(
        "--no-test", 
        action="store_true",
        help="Skip test search after population"
    )
    parser.add_argument(
        "--data-dir", 
        action="append",
        help="Additional data directory to scan (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Run population
    results = populate_vector_db(
        collection_name=args.collection,
        data_directories=args.data_dir,
        force_reprocess=args.force,
        test_search=not args.no_test
    )
    
    # Print results
    if results["success"]:
        print("\n🎉 Vector database population completed successfully!")
        if "vectors_added" in results:
            print(f"   📊 Vectors added: {results['vectors_added']}")
            print(f"   ⏱️ Processing time: {results['processing_time']:.2f}s")
    else:
        print(f"\n❌ Population failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)