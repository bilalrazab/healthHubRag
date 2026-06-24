import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. TEXT CHUNKING ENGINE
# ==========================================

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits long markdown text blocks into smaller overlapping segments 
    to preserve structural context during vector search.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward by the step size (chunk_size minus overlap)
        start += (chunk_size - chunk_overlap)
        
    return chunks

# ==========================================
# 2. CHROMADB VECTOR INGESTION PIPELINE
# ==========================================

def load_vector_store(json_path: str, chroma_db_dir: str, collection_name: str):
    # Ensure raw inputs exist before initializing database runtimes
    if not Path(json_path).exists():
        print(f"[!] Error: Structured target json not found at {json_path}")
        return

    print("[*] Initializing Persistent ChromaDB Client...")
    Path(chroma_db_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize the local database client
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

    # Use a standard production-ready local embedding function
    # Note: Can be easily swapped out for OpenAI or Gemini embedding functions if needed
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Create or fetch the text knowledge collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    with open(json_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # Extract our text content entries (populated from general, specialities, news, etc.)
    text_entries = master_data.get("text_content", [])
    print(f"[*] Processing semantic content pipeline for {len(text_entries)} documents...")

    documents_batch = []
    metadatas_batch = []
    ids_batch = []

    for entry in text_entries:
        url = entry.get("url", "")
        source_type = entry.get("type", "general")
        title = entry.get("title", "Untitled Document")
        body = entry.get("clean_body", "").strip()

        if not body:
            continue

        # Generate a stable deterministic base hash from the URL
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()

        # Split long content into production-sized chunks
        chunks = chunk_text(body, chunk_size=1000, chunk_overlap=200)

        for idx, chunk in enumerate(chunks):
            # Create a unique structural ID per text segment
            chunk_id = f"doc_{url_hash}_chk_{idx}"
            
            documents_batch.append(chunk)
            metadatas_batch.append({
                "url": url,
                "source_type": source_type,
                "title": title,
                "chunk_index": idx
            })
            ids_batch.append(chunk_id)

    # 3. SAFE BATCHED UPLOADING WRAPPER
    # Prevents HTTP or memory payload overflow bottlenecks
    batch_size = 200
    total_chunks = len(documents_batch)
    print(f"[+] Total text chunks generated: {total_chunks}. Commencing database push...")

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        
        # Slice current chunk batches
        docs = documents_batch[i:end_idx]
        metas = metadatas_batch[i:end_idx]
        ids = ids_batch[i:end_idx]

        # .upsert ensures safe rewriting if the pipeline runs multiple times
        collection.upsert(
            documents=docs,
            metadatas=metas,
            ids=ids
        )
        print(f"    -> Successfully indexed chunks {i} through {end_idx}")

    print(f"[✓] ChromaDB Collection '{collection_name}' fully populated and indexed.")


if __name__ == "__main__":
    INPUT_JSON = "./data/structured/structured_data.json"
    CHROMA_DIR = "./data/db/chroma"
    COLLECTION_NAME = "clinic_knowledge"

    load_vector_store(INPUT_JSON, CHROMA_DIR, COLLECTION_NAME)