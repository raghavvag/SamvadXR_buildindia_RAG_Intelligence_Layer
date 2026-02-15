"""
Samvad XR — RAG Operations (ChromaDB Knowledge Retrieval)
==========================================================
Provides cultural context for the LLM: bargaining norms,
item prices, and Hindi phrases.

Uses ChromaDB in-process (no external DB needed).
Default embedding: all-MiniLM-L6-v2 via sentence-transformers.
"""

import asyncio
import glob
import logging
import os
import textwrap
import uuid

import chromadb

from services.exceptions import RAGServiceError

logger = logging.getLogger("samvadxr.rag")

# ── Module-level state ─────────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

# Where seed data lives (relative to project root)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Chunking config
_CHUNK_SIZE_WORDS = 200
_CHUNK_OVERLAP_WORDS = 30


# ── Text chunking ─────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE_WORDS,
                overlap: int = _CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` words.

    Uses word-level splitting with `overlap` words carried over between
    consecutive chunks so semantic context isn't lost at boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap  # slide with overlap

    return chunks


# ── Initialization ─────────────────────────────────────────────────────

def initialize_knowledge_base(data_dir: str | None = None) -> None:
    """
    Load all .txt files from the data directory into ChromaDB.

    Creates an in-memory ChromaDB client and a collection called
    "samvad_context" with cosine similarity. Each file is chunked
    into ~200-word segments and added with source metadata.

    Called once by Dev B in the FastAPI lifespan startup event.
    """
    global _client, _collection

    data_path = data_dir or _DATA_DIR

    _client = chromadb.Client()
    _collection = _client.get_or_create_collection(
        name="samvad_context",
        metadata={"hnsw:space": "cosine"},
    )

    # Clear any stale data from previous runs (in-memory client reuse edge case)
    if _collection.count() > 0:
        all_ids = _collection.get()["ids"]
        if all_ids:
            _collection.delete(ids=all_ids)

    # Find all .txt files in the data directory
    txt_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))
    if not txt_files:
        logger.warning("No .txt files found in %s — knowledge base is empty", data_path)
        return

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        source_name = os.path.splitext(filename)[0]  # e.g. "bargaining_culture"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            logger.error("Failed to read %s: %s", filepath, e)
            continue

        chunks = _chunk_text(content)
        if not chunks:
            logger.warning("No content after chunking: %s", filename)
            continue

        for i, chunk in enumerate(chunks):
            doc_id = f"{source_name}_{i}"
            all_docs.append(chunk)
            all_ids.append(doc_id)
            all_metas.append({"source": filename, "chunk_index": i})

        logger.info("Loaded %s: %d chunks", filename, len(chunks))

    if all_docs:
        _collection.add(
            documents=all_docs,
            ids=all_ids,
            metadatas=all_metas,
        )
        logger.info(
            "Knowledge base ready: %d chunks from %d files",
            len(all_docs), len(txt_files),
        )
    else:
        logger.warning("Knowledge base is empty — no chunks were created")


# ── Retrieval ──────────────────────────────────────────────────────────

async def retrieve_context(query: str, n_results: int = 3) -> str:
    """
    Retrieve relevant cultural knowledge for a user query.

    Wraps ChromaDB's synchronous query in asyncio.to_thread()
    for non-blocking operation.

    Args:
        query:     The user's transcribed speech or a search term.
        n_results: Number of top chunks to return (default 3).

    Returns:
        A single string with the top N chunks joined by newlines,
        or "" if no results or knowledge base not initialized.

    Raises:
        RAGServiceError: If ChromaDB query fails unexpectedly.
    """
    if _collection is None:
        logger.warning("retrieve_context called before initialization")
        return ""

    if not query or not query.strip():
        return ""

    try:
        results = await asyncio.to_thread(
            _collection.query,
            query_texts=[query],
            n_results=n_results,
        )
    except Exception as e:
        raise RAGServiceError(detail=f"ChromaDB query failed: {e}") from e

    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return ""

    # Join the top chunks into a single context string
    context = "\n".join(doc for doc in documents[0] if doc)

    logger.info(
        "RAG retrieved %d chunks for query: \"%s\" (%d chars)",
        len(documents[0]), query[:50], len(context),
    )
    return context

