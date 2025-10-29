"""
Utility package for the mental health chatbot application.

This package contains various utility functions and scripts for:
- Database operations
- Vector database population
- Data processing
- System maintenance
"""

from .vector_db_populator import VectorDBPopulator, populate_vector_db

__all__ = [
    'VectorDBPopulator',
    'populate_vector_db'
]