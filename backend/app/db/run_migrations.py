#!/usr/bin/env python3
"""
Database Migration Runner

This script runs all database migrations in the correct order.
Migrations are designed to be idempotent - safe to run multiple times.

Usage:
    python3 backend/app/db/run_migrations.py
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Callable

# Add the project root to the Python path
# From backend/app/db/run_migrations.py -> go up 3 levels to project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.db.database import DATABASE_URL, engine

# Import migration functions
from backend.app.db.migration_add_gender import add_gender_column

# Migration registry: (name, function, description)
MIGRATIONS: List[Tuple[str, Callable[[], bool], str]] = [
    (
        "add_gender_column",
        add_gender_column,
        "Add gender column to users table for cultural personalization"
    ),
    # Add future migrations here in order
]


def run_all_migrations(dry_run: bool = False) -> bool:
    """
    Run all migrations in order.
    
    Args:
        dry_run: If True, only print what would be done without executing
        
    Returns:
        True if all migrations succeeded, False otherwise
    """
    print("=" * 60)
    print("🚀 Database Migration Runner")
    print("=" * 60)
    # Mask password in database URL for display
    try:
        if '@' in DATABASE_URL:
            parts = DATABASE_URL.split('@')
            db_display = parts[-1]  # Get everything after @
        else:
            db_display = DATABASE_URL
        print(f"📊 Database: {db_display}")
    except Exception:
        print(f"📊 Database: {DATABASE_URL[:50]}...")
    print(f"🔍 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)
    print()
    
    if dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made\n")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for migration_name, migration_func, description in MIGRATIONS:
        print(f"📝 Migration: {migration_name}")
        print(f"   {description}")
        
        if dry_run:
            print(f"   [DRY RUN] Would run: {migration_name}")
            print("   ✅ Skipped (dry run mode)")
            skipped_count += 1
        else:
            try:
                result = migration_func()
                if result:
                    print(f"   ✅ {migration_name} completed successfully")
                    success_count += 1
                else:
                    print(f"   ⚠️  {migration_name} was skipped (may already be applied)")
                    skipped_count += 1
            except Exception as e:
                print(f"   ❌ {migration_name} failed: {e}")
                failed_count += 1
                # Continue with other migrations even if one fails
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Migration Summary")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📦 Total: {len(MIGRATIONS)}")
    print("=" * 60)
    
    return failed_count == 0


def run_specific_migration(migration_name: str) -> bool:
    """
    Run a specific migration by name.
    
    Args:
        migration_name: Name of the migration to run
        
    Returns:
        True if migration succeeded, False otherwise
    """
    for name, func, description in MIGRATIONS:
        if name == migration_name:
            print(f"🚀 Running migration: {migration_name}")
            print(f"   {description}")
            try:
                result = func()
                if result:
                    print(f"✅ {migration_name} completed successfully")
                else:
                    print(f"⚠️  {migration_name} was skipped (may already be applied)")
                return True
            except Exception as e:
                print(f"❌ {migration_name} failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    print(f"❌ Migration '{migration_name}' not found")
    print(f"Available migrations: {', '.join([name for name, _, _ in MIGRATIONS])}")
    return False


def list_migrations():
    """List all available migrations."""
    print("📋 Available Migrations:")
    print("=" * 60)
    for i, (name, _, description) in enumerate(MIGRATIONS, 1):
        print(f"{i}. {name}")
        print(f"   {description}")
        print()
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--migration",
        type=str,
        help="Run a specific migration by name"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available migrations"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_migrations()
        sys.exit(0)
    
    if args.migration:
        success = run_specific_migration(args.migration)
        sys.exit(0 if success else 1)
    else:
        success = run_all_migrations(dry_run=args.dry_run)
        sys.exit(0 if success else 1)

