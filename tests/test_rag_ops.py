"""
Tests for services/rag_ops.py — ChromaDB knowledge retrieval.
Covers initialization, chunking, retrieval, edge cases, and errors.
"""

import os
import tempfile

import pytest

from services.rag_ops import (
    initialize_knowledge_base,
    retrieve_context,
    _chunk_text,
    _collection,
)
from services.exceptions import RAGServiceError


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_rag_state():
    """Reset module-level RAG state before each test."""
    import services.rag_ops as rag
    rag._client = None
    rag._collection = None
    yield
    rag._client = None
    rag._collection = None


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary data directory with test files."""
    items_file = tmp_path / "items.txt"
    items_file.write_text(
        "Tomato: Wholesale ₹20-30/kg, Fair Retail ₹40-60/kg, Tourist Price ₹100-150/kg. "
        "Firm and bright red means fresh. Summer prices lowest.\n\n"
        "Mango: Wholesale ₹40-80/kg, Fair Retail ₹80-150/kg, Tourist Price ₹200-400/kg. "
        "Alphonso is the premium variety. Available April-July only.\n\n"
        "Watermelon: Wholesale ₹8-12/kg, Fair Retail ₹15-25/kg, Tourist Price ₹40-60/kg. "
        "Knock for hollow sound to check ripeness. Summer fruit.",
        encoding="utf-8",
    )

    culture_file = tmp_path / "culture.txt"
    culture_file.write_text(
        "In Indian street markets, bargaining is a social ritual. "
        "The vendor's first price is always 2-4 times the actual value. "
        "A good buyer starts at 25-30% of the quoted price. "
        "Walking away is the most powerful bargaining lever.",
        encoding="utf-8",
    )

    phrases_file = tmp_path / "phrases.txt"
    phrases_file.write_text(
        '"Kitne ka hai?" (कितने का है?) — "How much is this?" — The universal opener.\n'
        '"Bahut mehnga hai!" (बहुत महंगा है!) — "Too expensive!" — Show shock.\n'
        '"Thoda kam karo" (थोड़ा कम करो) — "Reduce it a bit" — Polite negotiation.',
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def initialized_rag(sample_data_dir):
    """Initialize RAG with sample data and return the data dir path."""
    initialize_knowledge_base(data_dir=str(sample_data_dir))
    return sample_data_dir


# ── Text Chunking ─────────────────────────────────────────────────────

class TestChunkText:
    """Tests for the _chunk_text() helper."""

    def test_empty_string_returns_empty_list(self):
        assert _chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _chunk_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello world this is a short text."
        chunks = _chunk_text(text, chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        # 400 words → should produce more than 1 chunk with default 200-word size
        words = ["word"] * 400
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=200, overlap=30)
        assert len(chunks) >= 2

    def test_overlap_preserves_context(self):
        """Last words of chunk N should appear at start of chunk N+1."""
        words = [f"w{i}" for i in range(300)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        # The last 20 words of chunk 0 should appear at the start of chunk 1
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        overlap_tail = chunk0_words[-20:]
        overlap_head = chunk1_words[:20]
        assert overlap_tail == overlap_head

    def test_all_words_covered(self):
        """Every word in the original text should appear in at least one chunk."""
        words = [f"unique{i}" for i in range(250)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        for word in words:
            assert word in all_chunk_words


# ── Initialization ────────────────────────────────────────────────────

class TestInitializeKnowledgeBase:
    """Tests for initialize_knowledge_base()."""

    def test_initializes_with_sample_data(self, sample_data_dir):
        initialize_knowledge_base(data_dir=str(sample_data_dir))
        import services.rag_ops as rag
        assert rag._client is not None
        assert rag._collection is not None

    def test_collection_has_documents(self, sample_data_dir):
        initialize_knowledge_base(data_dir=str(sample_data_dir))
        import services.rag_ops as rag
        count = rag._collection.count()
        assert count > 0

    def test_empty_data_dir_creates_empty_collection(self, tmp_path):
        initialize_knowledge_base(data_dir=str(tmp_path))
        import services.rag_ops as rag
        assert rag._collection is not None
        assert rag._collection.count() == 0

    def test_loads_all_txt_files(self, sample_data_dir):
        """All 3 test files should contribute chunks."""
        initialize_knowledge_base(data_dir=str(sample_data_dir))
        import services.rag_ops as rag
        # Query the collection metadata to check sources
        all_data = rag._collection.get()
        sources = {m["source"] for m in all_data["metadatas"]}
        assert "items.txt" in sources
        assert "culture.txt" in sources
        assert "phrases.txt" in sources

    def test_uses_default_data_dir_if_none(self):
        """When no data_dir is passed, it uses the project's data/ folder."""
        initialize_knowledge_base()
        import services.rag_ops as rag
        assert rag._collection is not None
        # Should have loaded from project data/ dir (which has our 3 seed files)
        assert rag._collection.count() > 0


# ── Retrieval ─────────────────────────────────────────────────────────

class TestRetrieveContext:
    """Tests for retrieve_context()."""

    @pytest.mark.asyncio
    async def test_returns_string(self, initialized_rag):
        result = await retrieve_context("tomato price")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tomato_query_returns_relevant_content(self, initialized_rag):
        result = await retrieve_context("tomato price")
        assert len(result) > 0
        # Should contain price-related content
        assert "₹" in result or "Tomato" in result or "tomato" in result.lower()

    @pytest.mark.asyncio
    async def test_bargaining_query_returns_culture_content(self, initialized_rag):
        result = await retrieve_context("how to bargain in Indian market")
        assert len(result) > 0
        assert "bargain" in result.lower() or "vendor" in result.lower() or "price" in result.lower()

    @pytest.mark.asyncio
    async def test_hindi_phrase_query(self, initialized_rag):
        result = await retrieve_context("how to say too expensive in Hindi")
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_multiple_chunks_joined(self, initialized_rag):
        """With n_results=3, result may contain content from multiple chunks."""
        result = await retrieve_context("bazaar shopping tips", n_results=3)
        assert isinstance(result, str)
        # Should have substantial content from multiple chunks
        assert len(result) > 50

    @pytest.mark.asyncio
    async def test_n_results_limits_output(self, initialized_rag):
        result_1 = await retrieve_context("price", n_results=1)
        result_3 = await retrieve_context("price", n_results=3)
        # More results requested → generally longer or equal output
        assert len(result_1) <= len(result_3)

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, initialized_rag):
        result = await retrieve_context("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, initialized_rag):
        result = await retrieve_context("   ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_before_initialization(self):
        """Before initialize_knowledge_base(), should return ""."""
        result = await retrieve_context("anything")
        assert result == ""


# ── Error Handling ────────────────────────────────────────────────────

class TestRAGErrorHandling:
    """Test error scenarios."""

    @pytest.mark.asyncio
    async def test_corrupted_collection_raises_rag_error(self, initialized_rag):
        """If the collection's query method fails, RAGServiceError is raised."""
        import services.rag_ops as rag
        from unittest.mock import patch

        with patch.object(rag._collection, "query", side_effect=RuntimeError("DB corrupt")):
            with pytest.raises(RAGServiceError) as exc_info:
                await retrieve_context("test query")
            assert "ChromaDB query failed" in str(exc_info.value)
            assert "DB corrupt" in exc_info.value.detail


# ── Integration: Real Seed Data ───────────────────────────────────────

class TestRealSeedData:
    """Tests using the actual project seed data files in data/."""

    @pytest.fixture(autouse=True)
    def _init_real_data(self, _reset_rag_state):
        initialize_knowledge_base()

    @pytest.mark.asyncio
    async def test_tomato_pricing(self):
        result = await retrieve_context("What is a fair price for tomatoes?")
        assert "₹" in result

    @pytest.mark.asyncio
    async def test_walk_away_strategy(self):
        result = await retrieve_context("What happens if I walk away from the vendor?")
        assert "walk" in result.lower() or "away" in result.lower()

    @pytest.mark.asyncio
    async def test_hindi_greeting(self):
        result = await retrieve_context("How do I greet the vendor in Hindi?")
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_watermelon_freshness(self):
        result = await retrieve_context("How to check if watermelon is fresh and ripe?")
        assert "watermelon" in result.lower() or "knock" in result.lower() or "ripe" in result.lower() or "tarbooz" in result.lower()
