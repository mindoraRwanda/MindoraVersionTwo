#!/usr/bin/env python3
"""
Vector Database Query Script

Simple script to query the populated vector database.

Usage:
    python query_vector_db.py                           # Interactive mode
    python query_vector_db.py -q "depression treatment" # Single query
    python query_vector_db.py --stats                   # Show statistics
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Set environment
os.environ["ENVIRONMENT"] = "development"

from app.utils.vector_db_manager import main

if __name__ == "__main__":
    main()