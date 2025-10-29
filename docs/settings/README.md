# Settings documentation (moved)

The detailed architecture, implementation, migration and testing documents for the settings subsystem have been moved out of the runtime package to keep imports lean and reduce cognitive overhead.

Moved files (original locations):
- [`backend/app/settings/ARCHITECTURE_SUMMARY.md`](backend/app/settings/ARCHITECTURE_SUMMARY.md:1)
- [`backend/app/settings/ENHANCEMENT_REVIEW.md`](backend/app/settings/ENHANCEMENT_REVIEW.md:1)
- [`backend/app/settings/IMPLEMENTATION_GUIDE.md`](backend/app/settings/IMPLEMENTATION_GUIDE.md:1)
- [`backend/app/settings/MIGRATION_GUIDE.md`](backend/app/settings/MIGRATION_GUIDE.md:1)
- [`backend/app/settings/TESTING_STRATEGY.md`](backend/app/settings/TESTING_STRATEGY.md:1)
- [`backend/app/settings/README.md`](backend/app/settings/README.md:1)

Runtime package entrypoint (use this when loading settings at runtime):
- [`backend/app/settings/__init__.py`](backend/app/settings/__init__.py:1)

If you need to reference design decisions, migration steps, or testing guidance, consult the files in this folder. The runtime package will keep only the Python modules required for actual configuration loading and validation.