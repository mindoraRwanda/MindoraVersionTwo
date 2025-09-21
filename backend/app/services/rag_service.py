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
PDF_FOLDER = Path(r"C:\Users\STUDENT\OneDrive\Desktop\CMU Essentials 2024-2026\Fall 2025 Courses\Capstone Project\mindora-conv-therapy-module\backend\datasources")
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
        full_text = "\n".join([page.get_text() for page in doc])
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

def initialize_rag_service():
    """Initializes the RAG service by processing PDF files and loading them into Qdrant."""
    try:
        # Load the sentence-transformer model
        logger.info("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dimension = embedder.get_sentence_embedding_dimension()

        # Initialize the Qdrant client
        logger.info("Connecting to Qdrant...")
        qdrant = QdrantClient(host="localhost", port=6333)

        # Create collection if it doesn't exist
        create_collection_if_not_exists(qdrant, COLLECTION_NAME, embedding_dimension)

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

# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams, Distance
# import uuid
# import fitz  # PyMuPDF


# # Define the folder containing your PDF documents
# # pdf_folder = Path("../../datasources")
# pdf_folder = Path(r"C:\Users\STUDENT\OneDrive\Desktop\CMU Essentials 2024-2026\Fall 2025 Courses\Capstone Project\mindora-conv-therapy-module\backend\datasources")

# # Define the name of the Qdrant collection to store document chunks
# collection_name = "therapy_knowledge_base"

# # Load the sentence-transformer model to convert text into dense embeddings
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize the Qdrant client to connect to the local vector database
# qdrant = QdrantClient(host="localhost", port=6333)


# # Initialize a text splitter to break long documents into smaller chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,      # Max characters per chunk
#     chunk_overlap=50     # Overlap to preserve context between chunks
# )

# # Print the PDF files found in the target folder
# print("Files found:", list(pdf_folder.glob("*.pdf")))

# # Loop through each PDF file for processing
# for file in pdf_folder.glob("*.pdf"):
#     print(f"\nProcessing file: {file.name}")

#     try:
#         # Open and read all text from the PDF file
#         doc = fitz.open(file)
#         full_text = "\n".join([page.get_text() for page in doc])
#         doc.close()

#         # Split the full document text into smaller overlapping chunks
#         chunks = splitter.split_text(full_text)
#         print(f"Split into {len(chunks)} chunks")

#         # Generate embeddings (numerical vectors) for each chunk
#         embeddings = embedder.encode(chunks).tolist()

#         # Create Qdrant-compatible Point objects to store each chunk and vector
#         points = []
#         for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
#             unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file.name}_{i}"))
#             points.append(PointStruct(
#                 id=unique_id,
#                 vector=vector,
#                 payload={
#                     "source": file.name,
#                     "chunk_id": i,
#                     "text": chunk
#                 }
#             ))

#         # Upload the chunks with embeddings and metadata to Qdrant
#         qdrant.upsert(
#             collection_name=collection_name,
#             points=points
#         )
#         print(f"Uploaded {len(points)} chunks to Qdrant.")

#     except Exception as e:
#         print(f"Error processing {file.name}: {e}")
        
    
    
    
