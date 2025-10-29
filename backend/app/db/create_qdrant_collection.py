from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
name = "therapy_knowledge_base"

try:
    client.get_collection(name)
    print(f"ℹ️ Collection '{name}' already exists")
except Exception:
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Created collection '{name}' with size=384, distance=Cosine")
