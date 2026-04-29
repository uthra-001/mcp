"""
rag.py — RAG (Retrieval-Augmented Generation) Engine
Handles: document loading, chunking, embedding, FAISS indexing, retrieval
"""

import re
import time
import pickle
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    import numpy as np
    print("[RAG] sentence-transformers/faiss not found — using keyword fallback.")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Streamlit Caching ─────────────────────────────────────────────────────────
try:
    import streamlit as st
    
    @st.cache_resource
    def _get_model():
        """Cache embedding model to avoid reloading."""
        if EMBEDDINGS_AVAILABLE:
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return None
    
    _get_cached_model = _get_model
except ImportError:
    def _get_cached_model():
        """Fallback without Streamlit caching."""
        if EMBEDDINGS_AVAILABLE:
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return None

# ── Globals ──────────────────────────────────────────────────────────────────
_model = None
_index = None
_chunks: list[str] = []
_chunk_sources: list[str] = []   # track which file each chunk came from

# ── Cache paths for fast loading ──────────────────────────────────────────────
CACHE_DIR = Path(".cache_rag")
CACHE_DIR.mkdir(exist_ok=True)
INDEX_CACHE_FILE = CACHE_DIR / "faiss_index.bin"
CHUNKS_CACHE_FILE = CACHE_DIR / "chunks.pkl"
SOURCES_CACHE_FILE = CACHE_DIR / "sources.pkl"

def _get_cache_hash(docs: list[tuple[str, str]]) -> str:
    """Generate hash of document contents to invalidate cache if docs change."""
    import hashlib
    content = "".join(f"{name}:{text}" for name, text in docs)
    return hashlib.md5(content.encode()).hexdigest()[:8]


# ── 1. Document Loading ───────────────────────────────────────────────────────

def load_documents(data_dir: str = "data") -> list[tuple[str, str]]:
    """
    Load all .txt and .pdf files from data_dir.
    Returns list of (filename, text) tuples.
    """
    results = []
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for file in sorted(data_path.iterdir()):
        if file.suffix == ".txt":
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                results.append((file.name, text))

        elif file.suffix == ".pdf" and PDF_AVAILABLE:
            with pdfplumber.open(file) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
                if text:
                    results.append((file.name, text))

    return results


# ── 2. Chunking ───────────────────────────────────────────────────────────────

def split_into_chunks(
    documents: list[tuple[str, str]],
    chunk_size: int = 25,   # ~25 words = highly specific chunks
    overlap: int = 5,       # minimal overlap
) -> tuple[list[str], list[str]]:
    
    """Split documents into small, focused chunks.
    Smaller chunks = more precise retrieval = better accuracy.
    """
    chunks, sources = [], []

    for filename, text in documents:
        # Split into sentences first
        sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        current_words = []
        current_sents = []

        for sent in sentences:
            sent_words = sent.split()
            current_words.extend(sent_words)
            current_sents.append(sent)

            if len(current_words) >= chunk_size:
                chunk = " ".join(current_words).strip()
                if len(chunk.split()) >= 10:  # Reduced from 15
                    chunks.append(chunk)
                    sources.append(filename)
                # Keep minimal overlap
                current_words = current_words[-overlap:]
                current_sents = []

        # Flush remaining
        if current_words and len(current_words) >= 10:
            chunks.append(" ".join(current_words).strip())
            sources.append(filename)

    return chunks, sources


# ── 3. Embedding + FAISS Index ────────────────────────────────────────────────

def _save_cache(chunks: list[str], sources: list[str]) -> None:
    """Save chunks and sources to disk cache."""
    try:
        with open(CHUNKS_CACHE_FILE, "wb") as f:
            pickle.dump(chunks, f)
        with open(SOURCES_CACHE_FILE, "wb") as f:
            pickle.dump(sources, f)
    except Exception as e:
        print(f"[RAG] Cache save failed: {e}")

def _load_cache() -> tuple[list[str], list[str], faiss.Index]:
    """Load cached chunks, sources, and FAISS index. Returns None if cache invalid."""
    try:
        if INDEX_CACHE_FILE.exists() and CHUNKS_CACHE_FILE.exists():
            with open(CHUNKS_CACHE_FILE, "rb") as f:
                chunks = pickle.load(f)
            with open(SOURCES_CACHE_FILE, "rb") as f:
                sources = pickle.load(f)
            index = faiss.read_index(str(INDEX_CACHE_FILE))
            print(f"[RAG] ✅ Loaded from cache: {len(chunks)} chunks")
            return chunks, sources, index
    except Exception as e:
        print(f"[RAG] Cache load failed: {e}")
    return None, None, None

def build_index(chunks: list[str], sources: list[str]) -> None:
    """Embed chunks with all-MiniLM-L6-v2 and store in a FAISS L2 index."""
    global _model, _index, _chunks, _chunk_sources

    _chunks = chunks
    _chunk_sources = sources

    if not EMBEDDINGS_AVAILABLE or not chunks:
        return

    print(f"[RAG] Building index for {len(chunks)} chunks...")
    _model = _get_cached_model()  # Use cached model to avoid reloading
    if not _model:
        return
    
    t0 = time.perf_counter()
    embeddings = _model.encode(chunks, show_progress_bar=False, batch_size=64).astype("float32")
    faiss.normalize_L2(embeddings)
    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)
    
    # Save to cache for next load
    try:
        faiss.write_index(_index, str(INDEX_CACHE_FILE))
        _save_cache(chunks, sources)
    except Exception as e:
        print(f"[RAG] Index caching failed: {e}")
    
    duration_ms = (time.perf_counter() - t0) * 1000
    print(f"[RAG] Index ready. {_index.ntotal} vectors stored in {duration_ms:.0f}ms")


def initialize(data_dir: str = "data") -> int:
    """
    Full pipeline: load → chunk → embed → index.
    Returns number of chunks indexed.
    
    ⚡ OPTIMIZATION: Tries cache first. If cache exists and is valid,
    loads instantly without re-embedding (10-100x faster).
    """
    global _model, _index, _chunks, _chunk_sources
    
    from backend import SAMPLE_DATA  # avoid circular at module level

    docs = load_documents(data_dir)
    if not docs:
        print("[RAG] No documents found — loading built-in sample data.")
        docs = [("sample_data.txt", SAMPLE_DATA)]

    # ⚡ TRY CACHE FIRST (instant load)
    cached_chunks, cached_sources, cached_index = _load_cache()
    if cached_chunks is not None and cached_index is not None:
        _chunks = cached_chunks
        _chunk_sources = cached_sources
        _index = cached_index
        # Load model but don't rebuild index
        if EMBEDDINGS_AVAILABLE:
            _model = _get_cached_model()
        return len(_chunks)

    # 🔨 REBUILD: Documents changed or cache invalid
    print("[RAG] Cache miss → rebuilding index...")
    chunks, sources = split_into_chunks(docs)
    build_index(chunks, sources)
    return len(chunks)


# ── 4. Retrieval ──────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant chunks using FAISS or keyword fallback.
    Returns chunks with relevance scores.
    """
    if not _chunks:
        return []

    if EMBEDDINGS_AVAILABLE and _index is not None:
        q_vec = _model.encode([query]).astype("float32")
        faiss.normalize_L2(q_vec)
        fetch = min(top_k * 3, len(_chunks))
        scores, indices = _index.search(q_vec, fetch)

        # Lower threshold (0.3 instead of 0.5) to catch more relevant results
        # FAISS inner product scores range from -1 to 1, lower threshold = better recall
        results = [
            {
                "chunk": _chunks[idx],
                "source": _chunk_sources[idx],
                "score": float(scores[0][rank]),
            }
            for rank, idx in enumerate(indices[0])
            if idx < len(_chunks) and scores[0][rank] > 0.3
        ]
        return results[:top_k] if results else []

    # Keyword fallback - more lenient matching
    query_terms = set(query.lower().split())
    query_terms = {t for t in query_terms if len(t) > 2}  # Skip tiny words
    
    if not query_terms:
        return []
    
    scored = []
    for i, chunk in enumerate(_chunks):
        chunk_words = chunk.lower().split()
        chunk_set = set(chunk_words)
        
        # Calculate overlap percentage
        overlap = len(query_terms & chunk_set)
        coverage = overlap / len(query_terms) if query_terms else 0
        
        # More lenient: coverage >= 50% OR at least 2 keywords matched
        tf = sum(chunk_words.count(t) for t in query_terms)
        score = coverage * 100 + tf * 10
        
        if coverage >= 0.5 or overlap >= 2:
            scored.append((score, i))

    scored.sort(reverse=True)
    return [
        {"chunk": _chunks[i], "source": _chunk_sources[i], "score": float(s)}
        for s, i in scored[:top_k]
    ]