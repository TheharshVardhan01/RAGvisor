from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import uuid

def embed_and_store(chunks, persist_dir, collection_name=None, overwrite=False):
    """
    Embed text chunks and store them in ChromaDB.
    
    Args:
        chunks: List of (text, metadata) tuples.
        persist_dir: Path to save ChromaDB DB.
        collection_name: Optional. Defaults to 'rag_pdf_<uuid>'.
        overwrite: If True, clears collection before inserting.
    
    Returns:
        collection_name used
    """
    if not collection_name:
        collection_name = f"rag_pdf_{str(uuid.uuid4())[:8]}"

    try:
        # Initialize model & Chroma client
        model = SentenceTransformer("all-MiniLM-L6-v2")
        client = PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(name=collection_name)

        if overwrite:
            collection.delete(where={})  # Clear old entries

        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        ids = [meta["chunk_id"] for meta in metadatas]

        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return collection_name

    except Exception as e:
        raise RuntimeError(f"[EMBED FAIL] {e}")
