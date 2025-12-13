import sys
import os

print("Testing imports...")

try:
    print("Importing langchain_core.messages...")
    from langchain_core.messages import BaseMessage
    print("✅ BaseMessage imported")
    from langchain_core.messages import HumanMessage
    print("✅ HumanMessage imported")
    from langchain_core.messages import SystemMessage
    print("✅ SystemMessage imported")
    from langchain_core.messages import AIMessage
    print("✅ AIMessage imported")
except Exception as e:
    print(f"❌ langchain_core.messages failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing pydantic...")
    from pydantic import BaseModel
    print("✅ pydantic imported")
except Exception as e:
    print(f"❌ pydantic failed: {e}")

try:
    print("Importing langchain_community.llms...")
    from langchain_community.llms import HuggingFacePipeline
    print("✅ langchain_community.llms imported")
except Exception as e:
    print(f"❌ langchain_community.llms failed: {e}")

try:
    print("Importing app.services.llm_providers...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import app.services.llm_providers
    print("✅ app.services.llm_providers imported")
except Exception as e:
    print(f"❌ app.services.llm_providers failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing app.db.database...")
    from app.db.database import SessionLocal
    print("✅ app.db.database imported")
except Exception as e:
    print(f"❌ app.db.database failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing app.settings.settings...")
    from app.settings.settings import settings
    print("✅ app.settings.settings imported")
except Exception as e:
    print(f"❌ app.settings.settings failed: {e}")
    import traceback
    traceback.print_exc()
