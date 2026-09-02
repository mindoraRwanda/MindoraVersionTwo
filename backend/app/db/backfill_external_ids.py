#!/usr/bin/env python3
"""
Backfill `users.external_id` from a main-app user export, so existing chatbot
users keep their conversations, emotion logs and crisis history after the
integration goes live.

This is optional. Without it, accounts are claimed lazily on first chat request
(see INTEGRATION_AUTO_LINK_BY_EMAIL in auth/integration_auth.py). Run it when you would
rather do the linking up front and see the conflicts before users do.

Input: a CSV exported from the main app's database, with a header row:

    id,email
    41,alice@example.com
    42,bob@example.com

Export it on the Node side, e.g.:
    COPY (SELECT id, lower(email) FROM users) TO STDOUT WITH CSV HEADER;

Usage:
    py -3.11 backend/app/db/backfill_external_ids.py main_app_users.csv --dry-run
    py -3.11 backend/app/db/backfill_external_ids.py main_app_users.csv --apply
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_dir)

from app.db.database import SessionLocal  # noqa: E402
from app.db.models import User  # noqa: E402


def load_main_app_users(path: str):
    by_email = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            email = (row.get("email") or "").strip().lower()
            user_id = (row.get("id") or "").strip()
            if email and user_id:
                by_email[email].append(user_id)
    return by_email


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV export of main-app users (id,email)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    group.add_argument("--apply", action="store_true", help="write external_id values")
    args = parser.parse_args()

    by_email = load_main_app_users(args.csv_path)
    print(f"Loaded {len(by_email)} main-app emails from {args.csv_path}")

    db = SessionLocal()
    linked = skipped_already = no_match = ambiguous = conflict = 0
    try:
        for user in db.query(User).all():
            email = (user.email or "").strip().lower()

            if user.external_id:
                skipped_already += 1
                continue

            candidates = by_email.get(email, [])
            if not candidates:
                no_match += 1
                print(f"  [no match]   chatbot user {user.id} <{email}> — will keep its history "
                      f"only if that person later signs in with the same email")
                continue
            if len(candidates) > 1:
                ambiguous += 1
                print(f"  [AMBIGUOUS]  {email} maps to {len(candidates)} main-app ids {candidates} — skipped")
                continue

            external_id = candidates[0]
            taken = db.query(User).filter(User.external_id == external_id).first()
            if taken:
                conflict += 1
                print(f"  [CONFLICT]   main-app id {external_id} already linked to chatbot user {taken.id} — skipped")
                continue

            if args.apply:
                user.external_id = external_id
                user.auth_provider = "mindora_web"
            linked += 1
            print(f"  [link]       chatbot user {user.id} <{email}> -> main-app {external_id}")

        if args.apply:
            db.commit()
            print("\nChanges committed.")
        else:
            db.rollback()
            print("\nDry run — nothing was written.")
    finally:
        db.close()

    print(
        f"\nSummary: {linked} linked, {skipped_already} already linked, "
        f"{no_match} unmatched, {ambiguous} ambiguous, {conflict} conflicting"
    )
    if ambiguous or conflict:
        print("Resolve AMBIGUOUS/CONFLICT rows by hand before turning off the standalone login.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
