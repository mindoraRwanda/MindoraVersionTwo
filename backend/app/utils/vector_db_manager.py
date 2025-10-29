"""
Vector Database Management Utility

This module provides utilities to manage and query the populated vector database.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.unified_rag_service import UnifiedRAGService


class VectorDBManager:
    """
    Utility class for managing the populated vector database.
    
    Provides functionality to:
    - Query the database
    - Get collection statistics
    - Test search functionality
    - Manage collections
    """
    
    def __init__(self, collection_name: str = "therapy_knowledge_base"):
        """Initialize the Vector DB Manager."""
        self.collection_name = collection_name
        self.rag_service = None
        
    def initialize(self) -> bool:
        """Initialize the RAG service."""
        try:
            self.rag_service = UnifiedRAGService(collection_name=self.collection_name)
            return self.rag_service.initialize(vectorize=False)
        except Exception as e:
            print(f"❌ Error initializing RAG service: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.rag_service:
            return {"error": "RAG service not initialized"}
        return self.rag_service.get_collection_info()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector database."""
        if not self.rag_service:
            return []
        return self.rag_service.search(query, top_k=top_k)
    
    def interactive_search(self):
        """Interactive search interface."""
        print("🔍 Interactive Vector Database Search")
        print("=" * 50)
        print("Enter your queries (type 'quit' to exit, 'stats' for collection info)")
        print()
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"📊 Collection: {stats.get('name', 'Unknown')}")
                    print(f"📊 Total vectors: {stats.get('vectors_count', 0) or 0:,}")
                    print(f"📊 Total points: {stats.get('points_count', 0) or 0:,}")
                    print(f"📊 Status: {stats.get('status', 'Unknown')}")
                    print()
                    continue
                
                if not query:
                    continue
                
                print(f"🔍 Searching for: '{query}'")
                results = self.search(query, top_k=5)
                
                if not results:
                    print("❌ No results found")
                    print()
                    continue
                
                print(f"✅ Found {len(results)} results:")
                print("-" * 50)
                
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    source = result.get('source', 'Unknown')
                    text = result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', '')
                    
                    print(f"{i}. Score: {score:.3f} | Source: {source}")
                    print(f"   Text: {text}")
                    print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function for the vector database manager CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage and query the vector database")
    parser.add_argument(
        "--collection", 
        default="therapy_knowledge_base",
        help="Name of the Qdrant collection"
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to execute"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show collection statistics"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive search mode"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Set environment
    if not os.getenv("ENVIRONMENT"):
        os.environ["ENVIRONMENT"] = "development"
    
    # Initialize manager
    manager = VectorDBManager(collection_name=args.collection)
    
    if not manager.initialize():
        print("❌ Failed to initialize vector database manager")
        sys.exit(1)
    
    # Handle different modes
    if args.stats:
        stats = manager.get_stats()
        print("📊 Vector Database Statistics")
        print("=" * 40)
        print(f"Collection: {stats.get('name', 'Unknown')}")
        print(f"Total vectors: {stats.get('vectors_count', 0) or 0:,}")
        print(f"Total points: {stats.get('points_count', 0) or 0:,}")
        print(f"Indexed vectors: {stats.get('indexed_vectors_count', 0) or 0:,}")
        print(f"Status: {stats.get('status', 'Unknown')}")
        
    elif args.query:
        print(f"🔍 Searching for: '{args.query}'")
        results = manager.search(args.query, top_k=args.top_k)
        
        if not results:
            print("❌ No results found")
        else:
            print(f"✅ Found {len(results)} results:")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                source = result.get('source', 'Unknown')
                text = result.get('text', '')[:300] + '...' if len(result.get('text', '')) > 300 else result.get('text', '')
                
                print(f"{i}. Score: {score:.3f}")
                print(f"   Source: {source}")
                print(f"   Text: {text}")
                print()
    
    elif args.interactive:
        manager.interactive_search()
    
    else:
        # Default: show stats and start interactive mode
        stats = manager.get_stats()
        print("🎯 Vector Database Manager")
        print("=" * 40)
        print(f"Collection: {stats.get('name', 'Unknown')}")
        print(f"Total vectors: {stats.get('vectors_count', 0) or 0:,}")
        print(f"Status: {stats.get('status', 'Unknown')}")
        print()
        
        manager.interactive_search()


if __name__ == "__main__":
    main()