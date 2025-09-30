from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
import fitz  # PyMuPDF
import logging
import torch
import gc
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PDF_FOLDER = Path("./data")  # Default folder for PDF files
COLLECTION_NAME = "therapy_knowledge_base"
BATCH_SIZE = 16  # Process embeddings in smaller batches
MAX_CHUNKS_PER_FILE = 1000  # Limit chunks for very large files

def create_collection_if_not_exists(qdrant_client, collection_name, embedding_dimension):
    """Create Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
            )
            logger.info("Collection created successfully")
        else:
            logger.info("Collection already exists")
    except Exception as e:
        logger.error(f"Error with collection management: {e}")
        raise

def process_embeddings_in_batches(embedder, chunks, batch_size=16):
    """Process embeddings in batches to avoid memory issues"""
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing embeddings batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        try:
            # Generate embeddings for batch
            batch_embeddings = embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to prevent overheating/overloading
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Try with smaller batch size
            if batch_size > 4:
                logger.info("Retrying with smaller batch size...")
                return process_embeddings_in_batches(embedder, chunks, batch_size // 2)
            else:
                raise e
    
    return all_embeddings

def process_file(file_path, embedder, splitter, qdrant_client, collection_name):
    """Process a single PDF file"""
    try:
        logger.info(f"Processing file: {file_path.name}")
        
        # Check file size
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        if file_size > 50:
            logger.warning(f"File {file_path.name} is large ({file_size:.1f}MB). Processing may be slow.")
        
        # Open and read all text from the PDF file
        doc = fitz.open(file_path)
        full_text = "\n".join([page.get_text() for page in doc if hasattr(page, 'get_text')])
        doc.close()

        if not full_text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return False

        # Split the full document text into smaller overlapping chunks
        chunks = splitter.split_text(full_text)
        logger.info(f"Split into {len(chunks)} chunks")

        # Limit chunks for very large documents to avoid memory issues
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            logger.warning(f"Limiting chunks from {len(chunks)} to {MAX_CHUNKS_PER_FILE}")
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        # Generate embeddings in batches
        embeddings = process_embeddings_in_batches(embedder, chunks, BATCH_SIZE)

        # Create Qdrant-compatible Point objects to store each chunk and vector
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path.name}_{i}"))
            points.append(PointStruct(
                id=unique_id,
                vector=vector,
                payload={
                    "source": file_path.name,
                    "chunk_id": i,
                    "text": chunk
                }
            ))

        # Upload the chunks with embeddings and metadata to Qdrant in batches
        upload_batch_size = 100
        for i in range(0, len(points), upload_batch_size):
            batch_points = points[i:i + upload_batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            logger.info(f"Uploaded batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")

        logger.info(f"Successfully processed {file_path.name} - uploaded {len(points)} chunks")
        return True

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return False

class RAGService:
    """RAG Service for managing vector database operations and document processing."""

    def __init__(self):
        """Initialize the RAG service."""
        self.qdrant_client = None
        self.embedder = None
        self.collection_name = COLLECTION_NAME
        self._initialized = False

    def initialize(self, vectorize: bool = False) -> bool:
        """Initialize the RAG service by processing PDF files and loading them into Qdrant."""
        try:
            # Initialize the Qdrant client first
            logger.info("Connecting to Qdrant...")
            self.qdrant_client = QdrantClient(host="localhost", port=6333)

            # Load the sentence-transformer model
            logger.info("Loading embedding model...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_dimension = self.embedder.get_sentence_embedding_dimension()

            # Always create collection, regardless of vectorize flag
            logger.info(f"Ensuring collection '{self.collection_name}' exists...")
            create_collection_if_not_exists(self.qdrant_client, self.collection_name, embedding_dimension)
            logger.info(f"✅ Collection '{self.collection_name}' is ready")

            if not vectorize:
                logger.info("Vectorization skipped as per configuration, but collection is available")
                self._initialized = True
                return True

            # Initialize text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,      # Smaller chunks for better processing
                chunk_overlap=40     # Proportional overlap
            )

            # Get PDF files
            try:
                pdf_folder_path = Path(PDF_FOLDER)
                pdf_files = list(pdf_folder_path.glob("*.pdf")) if pdf_folder_path.exists() else []
                logger.info(f"Found {len(pdf_files)} PDF files")
            except Exception as e:
                logger.warning(f"Error accessing PDF folder: {e}")
                pdf_files = []

            successful = 0
            failed = 0

            # Process each file
            for i, file_path in enumerate(pdf_files, 1):
                logger.info(f"\n--- Processing file {i}/{len(pdf_files)} ---")

                try:
                    if process_file(file_path, self.embedder, splitter, self.qdrant_client, self.collection_name):
                        successful += 1
                    else:
                        failed += 1
                except KeyboardInterrupt:
                    logger.info("\nProcessing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")
                    failed += 1

            logger.info(f"\n=== Processing Complete ===")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Total: {successful + failed}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Fatal error during RAG service initialization: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if the RAG service is properly initialized."""
        return self._initialized and self.qdrant_client is not None

    def get_client(self) -> QdrantClient | None:
        """Get the Qdrant client instance."""
        return self.qdrant_client

    def search_similar(self, query: str, limit: int = 5) -> list:
        """Search for similar documents in the vector database."""
        if not self.is_initialized() or self.embedder is None or self.qdrant_client is None:
            logger.warning("RAG service not properly initialized")
            return []

        try:
            # Encode the query
            query_embedding = self.embedder.encode([query]).tolist()[0]

            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Format results
            results = []
            for point in search_result:
                payload = getattr(point, 'payload', {})
                results.append({
                    "id": getattr(point, 'id', ''),
                    "score": getattr(point, 'score', 0.0),
                    "text": payload.get("text", "") if payload else "",
                    "source": payload.get("source", "") if payload else "",
                    "chunk_id": payload.get("chunk_id", 0) if payload else 0
                })

            return results

        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []


def initialize_rag_service(vectorize: bool = False) -> QdrantClient | None:
    """Initializes the RAG service by processing PDF files and loading them into Qdrant."""
    try:
        # Initialize the Qdrant client first
        logger.info("Connecting to Qdrant...")
        qdrant = QdrantClient(host="localhost", port=6333)

        # Load the sentence-transformer model
        logger.info("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dimension = embedder.get_sentence_embedding_dimension()

        # Always create collection, regardless of vectorize flag
        logger.info(f"Ensuring collection '{COLLECTION_NAME}' exists...")
        create_collection_if_not_exists(qdrant, COLLECTION_NAME, embedding_dimension)
        logger.info(f"✅ Collection '{COLLECTION_NAME}' is ready")

        if not vectorize:
            logger.info("Vectorization skipped as per configuration, but collection is available")
            return qdrant

        

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,      # Smaller chunks for better processing
            chunk_overlap=40     # Proportional overlap
        )

        # Get PDF files
        pdf_files = list(PDF_FOLDER.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        successful = 0
        failed = 0

        # Process each file
        for i, file_path in enumerate(pdf_files, 1):
            logger.info(f"\n--- Processing file {i}/{len(pdf_files)} ---")
            
            try:
                if process_file(file_path, embedder, splitter, qdrant, COLLECTION_NAME):
                    successful += 1
                else:
                    failed += 1
            except KeyboardInterrupt:
                logger.info("\nProcessing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")
                failed += 1

        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {successful + failed}")
        
        return qdrant

    except Exception as e:
        logger.error(f"Fatal error during RAG service initialization: {e}")
        return None

if __name__ == "__main__":
    initialize_rag_service()
