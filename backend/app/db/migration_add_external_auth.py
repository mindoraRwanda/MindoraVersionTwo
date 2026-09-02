#!/usr/bin/env python3
"""
Database migration: link chatbot accounts to main web app accounts.

Adds two columns to `users`:

  external_id    The main app's user id. Unique, nullable — nullable because
                 pre-integration accounts have no main-app counterpart until
                 their owner signs in through the main app.
  auth_provider  'local' | 'google' | 'mindora_web'. Purely informational, but
                 it makes it obvious which accounts have been migrated.

Also relaxes NOT NULL on `username` is NOT done here — usernames are still
generated for every account. Password is already nullable from the Google
migration; this migration re-checks it so it can be run on a database that
never had that migration applied.

Run once per environment:
    py -3.11 backend/app/db/migration_add_external_auth.py
"""

import os
import sys

from sqlalchemy import text

backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_dir)

from app.db.database import engine  # noqa: E402


def _column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
            """
        ),
        {"table": table, "column": column},
    )
    return result.fetchone() is not None


def add_external_auth_columns() -> bool:
    try:
        with engine.connect() as conn:
            if _column_exists(conn, "users", "external_id"):
                print("[OK] external_id column already exists")
            else:
                print("Adding external_id column to users table...")
                conn.execute(text("ALTER TABLE users ADD COLUMN external_id VARCHAR(255)"))
                conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_external_id "
                        "ON users (external_id) WHERE external_id IS NOT NULL"
                    )
                )
                conn.commit()
                print("[OK] Successfully added external_id column and unique index")

            if _column_exists(conn, "users", "auth_provider"):
                print("[OK] auth_provider column already exists")
            else:
                print("Adding auth_provider column to users table...")
                conn.execute(
                    text("ALTER TABLE users ADD COLUMN auth_provider VARCHAR(20) DEFAULT 'local'")
                )
                conn.execute(
                    text(
                        "UPDATE users SET auth_provider = "
                        "CASE WHEN google_id IS NOT NULL THEN 'google' ELSE 'local' END "
                        "WHERE auth_provider IS NULL"
                    )
                )
                conn.commit()
                print("[OK] Successfully added auth_provider column")

            # password nullability — harmless if the Google migration already ran
            result = conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'users' AND column_name = 'password'"
                )
            )
            row = result.fetchone()
            if row and row[0] == "YES":
                print("[OK] password column is already nullable")
            else:
                print("Making password column nullable on users table...")
                conn.execute(text("ALTER TABLE users ALTER COLUMN password DROP NOT NULL"))
                conn.commit()
                print("[OK] Successfully made password column nullable")

        return True

    except Exception as e:
        print(f"[ERROR] Error migrating users table for external auth: {e}")
        return False


if __name__ == "__main__":
    ok = add_external_auth_columns()
    print("Migration completed successfully!" if ok else "Migration failed!")
    sys.exit(0 if ok else 1)
