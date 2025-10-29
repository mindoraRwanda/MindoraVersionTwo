#!/usr/bin/env python3
"""
Vector Database Population Script

Simple script to populate the vector database with documents for the mental health chatbot.

Usage:
    python populate_vector_db.py                    # Basic population
    python populate_vector_db.py --force            # Force reprocess all documents
    python populate_vector_db.py --help             # Show help
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Set environment
os.environ["ENVIRONMENT"] = "development"

from app.utils.vector_db_populator import populate_vector_db


def main():
    """Main function to run the vector database population."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Populate vector database with mental health documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python populate_vector_db.py                    # Basic population
    python populate_vector_db.py --force            # Force reprocess all documents
    python populate_vector_db.py --collection test  # Use different collection
    python populate_vector_db.py --data-dir ./docs  # Add custom data directory
        """
    )
    
    parser.add_argument(
        "--collection", 
        default="therapy_knowledge_base",
        help="Name of the Qdrant collection (default: therapy_knowledge_base)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing of all documents even if collection exists"
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Mental Health Chatbot - Vector Database Populator")
    print("=" * 60)
    
    # Run population
    try:
        results = populate_vector_db(
            collection_name=args.collection,
            data_directories=args.data_dir,
            force_reprocess=args.force,
            test_search=not args.no_test
        )
        
        # Print results summary
        print("\n" + "=" * 60)
        if results["success"]:
            print("🎉 SUCCESS: Vector database population completed!")
            
            if "vectors_added" in results:
                print(f"📊 Documents processed: {results.get('documents_found', 0)}")
                print(f"📊 Vectors added: {results['vectors_added']}")
                print(f"📊 Total vectors: {results.get('final_count', 0)}")
                print(f"⏱️  Processing time: {results['processing_time']:.2f} seconds")
                
                if "test_search_results" in results:
                    print(f"🔍 Test search: {results['test_search_results']} results found")
            else:
                print(f"ℹ️  {results.get('message', 'No processing needed')}")
                
        else:
            print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()