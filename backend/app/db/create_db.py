from backend.app.db.database import engine, Base
from backend.app.db import models  # <-- important: import models so Base knows them

print("Creating all tables in the database...")
Base.metadata.create_all(bind=engine)
print("✅ Done! Tables are ready.")