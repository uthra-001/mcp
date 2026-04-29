"""
Unit tests for rag.py — document loading, chunking, and retrieval.
Run with: pytest tests/test_rag.py -v
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rag import split_into_chunks, load_documents, retrieve, build_index, initialize


# ── Test: split_into_chunks ───────────────────────────────────────────────────
class TestSplitIntoChunks:
    def test_basic_chunking(self):
        # Use text with sentence boundaries so chunker can split properly
        text = ". ".join(["The project is delayed"] * 30) + "."
        docs = [("test.txt", text)]
        chunks, sources = split_into_chunks(docs, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_sources_match_chunks(self):
        docs = [("file_a.txt", "word " * 100), ("file_b.txt", "word " * 100)]
        chunks, sources = split_into_chunks(docs, chunk_size=50, overlap=10)
        assert len(chunks) == len(sources)

    def test_skips_tiny_fragments(self):
        docs = [("test.txt", "short")]
        chunks, sources = split_into_chunks(docs, chunk_size=100, overlap=10)
        # "short" is only 1 word — below min 15 words, should be skipped
        assert len(chunks) == 0

    def test_overlap_creates_more_chunks(self):
        text = "word " * 200
        docs = [("test.txt", text)]
        chunks_no_overlap, _ = split_into_chunks(docs, chunk_size=100, overlap=0)
        chunks_with_overlap, _ = split_into_chunks(docs, chunk_size=100, overlap=30)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_content_is_not_empty(self):
        docs = [("test.txt", "The project is delayed. Budget is over. Safety is good.")]
        chunks, _ = split_into_chunks(docs, chunk_size=5, overlap=1)
        assert all(len(c.strip()) > 0 for c in chunks)


# ── Test: load_documents ──────────────────────────────────────────────────────
class TestLoadDocuments:
    def test_loads_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "test.txt"
            txt_file.write_text("This is a test document.", encoding="utf-8")
            docs = load_documents(tmpdir)
            assert len(docs) == 1
            assert docs[0][0] == "test.txt"
            assert "test document" in docs[0][1]

    def test_returns_empty_for_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(tmpdir)
            assert docs == []

    def test_ignores_non_txt_non_pdf_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "notes.csv").write_text("a,b,c")
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")
            docs = load_documents(tmpdir)
            assert docs == []

    def test_creates_folder_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_data"
            docs = load_documents(str(new_dir))
            assert new_dir.exists()
            assert docs == []


# ── Test: retrieve (keyword fallback) ────────────────────────────────────────
class TestRetrieve:
    def setup_method(self):
        """Build a small in-memory index using keyword fallback."""
        import rag
        rag._chunks = [
            "The project is 6 weeks behind schedule due to delays.",
            "Budget consumed 91% of allocated funds.",
            "Safety compliance score is 94%.",
            "Foundation work is completed on time.",
            "Plumbing postponed due to permit issues.",
        ]
        rag._chunk_sources = ["report.pdf"] * 5
        rag._index = None  # force keyword fallback

    def test_returns_results(self):
        results = retrieve("delay schedule", top_k=3)
        assert len(results) > 0

    def test_result_has_required_keys(self):
        results = retrieve("budget", top_k=2)
        for r in results:
            assert "chunk" in r
            assert "source" in r
            assert "score" in r

    def test_top_k_limits_results(self):
        results = retrieve("project", top_k=2)
        assert len(results) <= 2

    def test_relevant_chunk_ranked_first(self):
        results = retrieve("budget funds", top_k=3)
        # The budget chunk should score highest
        assert "budget" in results[0]["chunk"].lower() or "funds" in results[0]["chunk"].lower()

    def test_empty_chunks_returns_empty(self):
        import rag
        rag._chunks = []
        results = retrieve("anything")
        assert results == []


# ── Test: initialize with sample data ────────────────────────────────────────
class TestInitialize:
    def test_initialize_with_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = initialize(data_dir=tmpdir)
            assert n > 0  # should load sample data

    def test_initialize_with_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt = Path(tmpdir) / "project.txt"
            txt.write_text(
                "The project is delayed by 4 weeks. Budget is 85% consumed. "
                "Foundation is complete. Electrical work is on schedule. "
                "Plumbing is postponed. Safety score is 94 percent. "
                "The revised completion date is March 2025. "
                "Material costs are 20% above estimates. "
                "Two contractors withdrew from the project. "
                "Concrete quality on floors 8 to 12 needs re-inspection.",
                encoding="utf-8",
            )
            n = initialize(data_dir=tmpdir)
            assert n > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
