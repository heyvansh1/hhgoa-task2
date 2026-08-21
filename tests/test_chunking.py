"""Unit tests for all four chunking strategies.

Covers: PassageNativeChunker, FixedSizeChunker, SemanticChunker,
HierarchicalChunker — including metadata population and edge cases.
"""

import pytest
from data.preprocess import Passage
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SemanticChunker
from chunking.hierarchical import HierarchicalChunker
from chunking.interface import BaseChunker


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_passages():
    return [
        Passage(
            passage_id="p1",
            text="This is a short test passage.",
            language="eng",
            query_cluster="q1",
        ),
        Passage(
            passage_id="p2",
            text=" ".join([f"word{i}" for i in range(100)]),
            language="eng",
            query_cluster="q1",
        ),
        Passage(
            passage_id="p3",
            text="",
            language="hin",
            query_cluster="q2",
        ),
    ]


@pytest.fixture
def multi_sentence_passage():
    """A passage with multiple clear sentences — useful for semantic chunking."""
    return Passage(
        passage_id="p_multi",
        text=(
            "The capital of India is New Delhi. "
            "It was designed by Edwin Lutyens. "
            "The speed of light is approximately 300000 km per second. "
            "Einstein described this in his theory of relativity."
        ),
        language="eng",
        query_cluster="q_multi",
    )


@pytest.fixture
def hindi_passage():
    return Passage(
        passage_id="p_hi",
        text="भारत की राजधानी नई दिल्ली है। मुंबई भारत की आर्थिक राजधानी है।",
        language="hin",
        query_cluster="q_hi",
    )


# ─────────────────────────────────────────────────────────────────────
# Baseline: PassageNativeChunker
# ─────────────────────────────────────────────────────────────────────

class TestPassageNativeChunker:
    def test_one_to_one(self, sample_passages):
        chunker = PassageNativeChunker()
        chunks = chunker.chunk(sample_passages)
        assert len(chunks) == 3
        assert chunks[0].chunk_id == "pn_p1_0"
        assert chunks[0].source_passage_id == "p1"
        assert chunks[0].language == "eng"
        assert chunks[0].query_cluster == "q1"
        assert chunks[0].text == "This is a short test passage."

    def test_metadata_populated(self, sample_passages):
        chunker = PassageNativeChunker()
        chunks = chunker.chunk(sample_passages)
        assert chunks[0].metadata["strategy"] == "passage_native"
        assert "char_length" in chunks[0].metadata

    def test_inherits_base(self):
        assert issubclass(PassageNativeChunker, BaseChunker)


# ─────────────────────────────────────────────────────────────────────
# Baseline: FixedSizeChunker
# ─────────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_short_passage(self, sample_passages):
        chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
        chunks = chunker.chunk([sample_passages[0]])
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short test passage."

    def test_long_passage_splits(self, sample_passages):
        chunker = FixedSizeChunker(chunk_size=30, overlap_percent=0.2)
        chunks = chunker.chunk([sample_passages[1]])
        assert len(chunks) > 1
        assert chunks[0].chunk_id == "fs_p2_0"
        assert len(chunks[0].text.split()) == 30

    def test_empty_passage(self, sample_passages):
        chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
        chunks = chunker.chunk([sample_passages[2]])
        assert len(chunks) == 0

    def test_metadata_populated(self, sample_passages):
        chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
        chunks = chunker.chunk([sample_passages[0]])
        assert chunks[0].metadata["strategy"] == "fixed_size"
        assert "chunk_index" in chunks[0].metadata
        assert "token_length" in chunks[0].metadata

    def test_inherits_base(self):
        assert issubclass(FixedSizeChunker, BaseChunker)


# ─────────────────────────────────────────────────────────────────────
# Advanced: SemanticChunker
# ─────────────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def test_produces_chunks(self, multi_sentence_passage):
        chunker = SemanticChunker(threshold_percentile=50.0)
        chunks = chunker.chunk([multi_sentence_passage])
        assert len(chunks) >= 1
        # All chunk text should be non-empty
        for c in chunks:
            assert c.text.strip()

    def test_metadata_fields(self, multi_sentence_passage):
        chunker = SemanticChunker()
        chunks = chunker.chunk([multi_sentence_passage])
        for c in chunks:
            assert c.metadata["strategy"] == "semantic"
            assert "embedding_model" in c.metadata
            assert "num_sentences" in c.metadata
            assert "char_length" in c.metadata

    def test_single_sentence(self, sample_passages):
        chunker = SemanticChunker()
        chunks = chunker.chunk([sample_passages[0]])
        assert len(chunks) == 1
        assert chunks[0].metadata["num_sentences"] == 1

    def test_hindi_passage(self, hindi_passage):
        chunker = SemanticChunker()
        chunks = chunker.chunk([hindi_passage])
        assert len(chunks) >= 1
        assert all(c.language == "hin" for c in chunks)

    def test_empty_passage(self, sample_passages):
        chunker = SemanticChunker()
        chunks = chunker.chunk([sample_passages[2]])
        assert len(chunks) == 0

    def test_inherits_base(self):
        assert issubclass(SemanticChunker, BaseChunker)


# ─────────────────────────────────────────────────────────────────────
# Advanced: HierarchicalChunker
# ─────────────────────────────────────────────────────────────────────

class TestHierarchicalChunker:
    def test_parent_and_children(self, sample_passages):
        chunker = HierarchicalChunker(child_token_size=30, child_overlap_percent=0.2)
        chunks = chunker.chunk([sample_passages[1]])  # 100-word passage

        parents = [c for c in chunks if c.metadata.get("hierarchy_level") == "parent"]
        children = [c for c in chunks if c.metadata.get("hierarchy_level") == "child"]

        assert len(parents) == 1
        assert len(children) >= 2
        assert all(c.metadata["parent_id"] == parents[0].chunk_id for c in children)

    def test_short_passage_single_child(self, sample_passages):
        chunker = HierarchicalChunker(child_token_size=50)
        chunks = chunker.chunk([sample_passages[0]])

        parents = [c for c in chunks if c.metadata.get("hierarchy_level") == "parent"]
        children = [c for c in chunks if c.metadata.get("hierarchy_level") == "child"]

        assert len(parents) == 1
        assert len(children) == 1
        assert children[0].text == sample_passages[0].text

    def test_metadata_fields(self, sample_passages):
        chunker = HierarchicalChunker(child_token_size=30)
        chunks = chunker.chunk([sample_passages[1]])
        for c in chunks:
            assert c.metadata["strategy"] == "hierarchical"
            assert "language" in c.metadata
            assert "char_length" in c.metadata
            assert "token_length" in c.metadata

    def test_empty_passage(self, sample_passages):
        chunker = HierarchicalChunker()
        chunks = chunker.chunk([sample_passages[2]])
        assert len(chunks) == 0

    def test_inherits_base(self):
        assert issubclass(HierarchicalChunker, BaseChunker)
