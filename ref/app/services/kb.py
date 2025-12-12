"""Knowledge Base retrieval service."""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import get_settings
from app.utils.logging import write_detailed_log, now_iso

settings = get_settings()

# Optional embeddings / vector DB:
HAS_EMB = False
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models as qmodels
    HAS_EMB = True
except Exception:
    HAS_EMB = False

# Global state
KB_CARDS: List[Dict[str, Any]] = []
tfidf = None
EMB = None
QDR = None


def load_kb_cards() -> List[Dict[str, Any]]:
    """Load KB cards from JSONL files."""
    cards = []
    kb_dir = Path(settings.KB_DIR)
    if kb_dir.exists():
        for p in kb_dir.glob("*.jsonl"):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        cards.append(json.loads(line))
                    except Exception:
                        pass
    return cards


def initialize_kb():
    """Initialize KB retrieval (TF-IDF and optionally embeddings)."""
    global KB_CARDS, tfidf, EMB, QDR
    
    KB_CARDS = load_kb_cards()
    
    # Fit TF-IDF once at startup
    if KB_CARDS:
        tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        tfidf.fit([" ".join([
            c.get("title", ""),
            " ".join(c.get("tags", [])),
            c.get("when_to_use", ""),
            c.get("bot_say", ""),
            " ".join(c.get("steps", []))
        ]) for c in KB_CARDS])
    
    # Optional vector DB with local Qdrant
    if HAS_EMB and settings.QDRANT_LOCAL_PATH:
        try:
            EMB = SentenceTransformer(settings.EMB_MODEL_NAME)
            QDR = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
            # Create collection if not exists
            dim = EMB.get_sentence_embedding_dimension()
            try:
                QDR.get_collection(settings.QDRANT_COLLECTION)
            except Exception:
                QDR.recreate_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
                )
            # Upsert KB if empty
            count = QDR.count(settings.QDRANT_COLLECTION).count
            if count == 0 and KB_CARDS:
                vecs = EMB.encode([c.get("bot_say", "") for c in KB_CARDS], normalize_embeddings=True)
                QDR.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=[qmodels.PointStruct(id=i, vector=vec.tolist(), payload=KB_CARDS[i])
                            for i, vec in enumerate(vecs)]
                )
        except Exception as e:
            print(f"Vector store init failed (fallback to TF-IDF): {e}")


def retrieve_kb(query: str, k: int = 2) -> List[Dict[str, Any]]:
    """Retrieve KB cards using TF-IDF."""
    if not KB_CARDS or not tfidf:
        return []
    docs = [" ".join([
        c.get("title", ""),
        " ".join(c.get("tags", [])),
        c.get("when_to_use", ""),
        c.get("bot_say", ""),
        " ".join(c.get("steps", []))
    ]) for c in KB_CARDS]
    qv = tfidf.transform([query])
    dv = tfidf.transform(docs)
    # Cosine (tf-idf vectors are L2-normalized by default)
    scores = (qv @ dv.T).toarray().ravel()
    idx = scores.argsort()[::-1][:k]
    return [KB_CARDS[i] for i in idx]


def retrieve_kb_semantic(query: str, k: int = 2) -> List[Dict[str, Any]]:
    """Retrieve KB cards using semantic search (Qdrant)."""
    if not (EMB and QDR and KB_CARDS):
        return []
    try:
        qv = EMB.encode([query], normalize_embeddings=True)[0].tolist()
        r = QDR.search(collection_name=settings.QDRANT_COLLECTION, query_vector=qv, limit=k)
        return [hit.payload for hit in r]
    except Exception:
        return []


def retrieve_kb_hybrid(
    query: str,
    k: int = 2,
    username: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Retrieve KB using TF-IDF first, fallback to semantic, with logging.
    
    Returns:
        Tuple of (kb_hits, method, time_seconds)
    """
    start_time = time.time()
    kb_hits = retrieve_kb(query, k=k)
    kb_method = "tfidf"
    if not kb_hits:
        kb_hits = retrieve_kb_semantic(query, k=k)
        kb_method = "semantic"

    elapsed = time.time() - start_time
    cards_returned = len(kb_hits)

    # Log retrieval event for later analysis (matches CLI-style logs)
    write_detailed_log(
        {
            "type": "kb_retrieval",
            "timestamp": now_iso(),
            "query": query,
            "method": kb_method,
            "cards_returned": cards_returned,
            "time_seconds": round(elapsed, 3),
            "cards": [
                {"title": c.get("title"), "bot_say": c.get("bot_say")}
                for c in kb_hits
            ],
        },
        username=username,
        conversation_id=conversation_id,
    )

    return kb_hits, kb_method, elapsed

