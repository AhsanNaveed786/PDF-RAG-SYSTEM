import os
import uuid
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "rag-index"
DIMENSION = 384


def _get_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
    return Pinecone(api_key=api_key)


def create_index():
    """Create Pinecone index if it doesn't already exist."""
    pc = _get_client()
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{INDEX_NAME}' created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def get_index():
    """Return a connected Pinecone Index object."""
    try:
        pc = _get_client()
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error connecting to index: {e}")
        return None


def clear_index():
    """Delete all vectors from the index."""
    try:
        pc = _get_client()
        index = pc.Index(INDEX_NAME)
        try:
            index.delete(delete_all=True, namespace="")
            print(f"All vectors cleared from '{INDEX_NAME}'.")
            return
        except:
            pass
        try:
            index.delete(delete_all=True)
            print(f"All vectors cleared from '{INDEX_NAME}'.")
            return
        except:
            pass
        print(f"Index '{INDEX_NAME}' is already empty or not initialized.")
    except Exception as e:
        print(f"Error clearing index: {e}")


def upsert_vectors(index, chunks, embeddings, sources=None):
    """
    Upsert chunk text + embeddings into Pinecone.
    Args:
        index: Pinecone index object
        chunks: List of text chunks
        embeddings: List of embedding vectors
        sources: Optional list of source filenames for each chunk
    """
    if index is None:
        raise ValueError("Index is None. Cannot upsert vectors.")

    if sources is None:
        sources = ["unknown"] * len(chunks)

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {
                "text": chunk,
                "chunk_id": i,
                "source": sources[i] if i < len(sources) else "unknown"
            }
        })

    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

    print(f"{len(vectors)} vectors upserted successfully.")
    return True