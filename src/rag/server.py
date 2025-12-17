"""
Simple RAG server for testing purposes.
Provides basic document retrieval and augmented generation with vector embeddings.
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Simple RAG Server", version="0.2.0")

# In-memory document store
documents: List[dict] = []

# Lazy load embedding model
_embedding_model = None


def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Vector search unavailable."
            )
            _embedding_model = False
    return _embedding_model if _embedding_model is not False else None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2))


class Document(BaseModel):
    """Document model."""

    id: str
    content: str
    metadata: Optional[dict] = None


class Query(BaseModel):
    """Query model."""

    query: str
    top_k: int = 3
    use_vector: bool = True  # Use vector search by default


class SearchResult(BaseModel):
    """Search result model."""

    id: str
    content: str
    score: float
    metadata: Optional[dict] = None


@app.get("/")
async def root():
    """Root endpoint."""
    model = get_embedding_model()
    return {
        "name": "Simple RAG Server",
        "version": "0.2.0",
        "documents": len(documents),
        "vector_search_available": model is not None,
    }


@app.post("/documents")
async def add_document(doc: Document):
    """Add a document to the store with optional embedding."""
    # Check if document already exists
    for existing_doc in documents:
        if existing_doc["id"] == doc.id:
            raise HTTPException(status_code=400, detail="Document ID already exists")

    doc_dict = doc.model_dump()

    # Generate embedding if model available
    model = get_embedding_model()
    if model is not None:
        embedding = model.encode(doc.content)
        doc_dict["embedding"] = embedding.tolist()
        logger.info(f"Generated embedding for document: {doc.id}")

    documents.append(doc_dict)
    logger.info(f"Added document: {doc.id}")
    return {"status": "success", "id": doc.id, "has_embedding": "embedding" in doc_dict}


@app.get("/documents")
async def list_documents():
    """List all documents."""
    return {"documents": documents, "count": len(documents)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    global documents
    original_count = len(documents)
    documents = [doc for doc in documents if doc["id"] != doc_id]

    if len(documents) == original_count:
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info(f"Deleted document: {doc_id}")
    return {"status": "success", "id": doc_id}


@app.delete("/documents")
async def clear_documents():
    """Clear all documents."""
    global documents
    count = len(documents)
    documents = []
    logger.info(f"Cleared {count} documents")
    return {"status": "success", "cleared": count}


@app.post("/search")
async def search(query: Query) -> dict:
    """
    Search documents using vector similarity or keyword matching.
    """
    if not documents:
        return {"results": [], "query": query.query, "method": "none"}

    model = get_embedding_model()
    use_vector = query.use_vector and model is not None

    if use_vector:
        # Vector-based search
        query_embedding = model.encode(query.query)
        results = []

        for doc in documents:
            if "embedding" not in doc:
                continue

            doc_embedding = np.array(doc["embedding"])
            similarity = cosine_similarity(query_embedding, doc_embedding)

            results.append(
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "score": float(similarity),
                    "metadata": doc.get("metadata"),
                }
            )

        # Sort by similarity descending
        results.sort(key=lambda x: x["score"], reverse=True)
        search_method = "vector"
    else:
        # Keyword-based search fallback
        results = []
        query_terms = query.query.lower().split()

        for doc in documents:
            content_lower = doc["content"].lower()
            # Simple scoring: count matching terms
            score = sum(1 for term in query_terms if term in content_lower)

            if score > 0:
                results.append(
                    {
                        "id": doc["id"],
                        "content": doc["content"],
                        "score": score / len(query_terms),  # Normalize score
                        "metadata": doc.get("metadata"),
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        search_method = "keyword"

    # Return top_k results
    top_results = results[: query.top_k]

    logger.info(
        f"Search query: '{query.query}' returned {len(top_results)} results using {search_method}"
    )
    return {
        "results": top_results,
        "query": query.query,
        "total_found": len(results),
        "method": search_method,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    model = get_embedding_model()
    docs_with_embeddings = sum(1 for doc in documents if "embedding" in doc)
    return {
        "status": "healthy",
        "documents": len(documents),
        "documents_with_embeddings": docs_with_embeddings,
        "vector_search_available": model is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7001)
