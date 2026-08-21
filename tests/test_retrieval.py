"""Tests for the hybrid retrieval engine.

Covers: DenseRetriever, SparseRetriever, HybridRetriever, latency constraints,
metadata pre-filtering, and fusion correctness.
"""

import time
import pytest
import numpy as np
from data.preprocess import Passage
from chunking.passage_native import PassageNativeChunker
from chunking.types import Chunk
from retrieval.retrieval import (
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corpus_chunks():
    """Build a small corpus of chunks for retrieval tests."""
    passages = [
        Passage("p1", "The capital of India is New Delhi, located on the west bank of the Yamuna River.", "eng", "q1"),
        Passage("p2", "Mumbai is the financial capital of India and the most populous city.", "eng", "q1"),
        Passage("p3", "The speed of light in vacuum is approximately 300000 kilometres per second.", "eng", "q2"),
        Passage("p4", "Albert Einstein developed the theory of general relativity.", "eng", "q2"),
        Passage("p5", "Python is a widely used programming language created by Guido van Rossum.", "eng", "q3"),
        Passage("p6", "भारत की राजधानी नई दिल्ली है, जो यमुना नदी के पश्चिमी तट पर स्थित है।", "hin", "q1"),
        Passage("p7", "मुंबई भारत की आर्थिक राजधानी है।", "hin", "q1"),
        Passage("p8", "निर्वात में प्रकाश की गति लगभग 300000 किलोमीटर प्रति सेकंड है।", "hin", "q2"),
        Passage("p9", "The Great Wall of China is the longest wall in the world.", "eng", "q4"),
        Passage("p10", "Machine learning is a subset of artificial intelligence focused on data-driven models.", "eng", "q5"),
    ]
    chunker = PassageNativeChunker()
    return chunker.chunk(passages)


# ─────────────────────────────────────────────────────────────────────
# Dense retriever
# ─────────────────────────────────────────────────────────────────────

class TestDenseRetriever:
    def test_build_and_search(self, corpus_chunks):
        from sentence_transformers import SentenceTransformer

        dr = DenseRetriever()
        dr.build(corpus_chunks)

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        q_emb = model.encode(["What is the capital of India?"], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)

        results = dr.search(q_emb[0], k=3)
        assert len(results) >= 1
        # The top result should be about India's capital
        top_ids = [r.chunk.chunk_id for r in results]
        assert any("p1" in cid or "p6" in cid for cid in top_ids)


# ─────────────────────────────────────────────────────────────────────
# Sparse retriever
# ─────────────────────────────────────────────────────────────────────

class TestSparseRetriever:
    def test_build_and_search(self, corpus_chunks):
        sr = SparseRetriever()
        sr.build(corpus_chunks)
        results = sr.search("capital India", k=3)
        assert len(results) >= 1
        top_ids = [r.chunk.chunk_id for r in results]
        assert any("p1" in cid or "p2" in cid for cid in top_ids)


# ─────────────────────────────────────────────────────────────────────
# Hybrid retriever
# ─────────────────────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_retrieve_english(self, corpus_chunks):
        retriever = HybridRetriever(corpus_chunks, fusion="rrf")
        results = retriever.retrieve("What is the capital of India?", k=3, lang="eng")
        assert len(results) >= 1
        assert all(isinstance(c, Chunk) for c in results)
        # Should find the India-capital passage
        top_ids = [c.chunk_id for c in results]
        assert any("p1" in cid for cid in top_ids)

    def test_retrieve_hindi(self, corpus_chunks):
        retriever = HybridRetriever(corpus_chunks, fusion="rrf")
        results = retriever.retrieve("भारत की राजधानी", k=3, lang="hin")
        assert len(results) >= 1
        assert all(c.language == "hin" for c in results)

    def test_retrieve_all_languages(self, corpus_chunks):
        retriever = HybridRetriever(corpus_chunks, fusion="rrf")
        results = retriever.retrieve("capital of India", k=5)
        assert len(results) >= 1
        # At least English should be present
        langs = {c.language for c in results}
        assert "eng" in langs

    def test_weighted_fusion(self, corpus_chunks):
        retriever = HybridRetriever(corpus_chunks, fusion="weighted", dense_weight=0.7)
        results = retriever.retrieve("speed of light", k=3)
        assert len(results) >= 1
        top_ids = [c.chunk_id for c in results]
        assert any("p3" in cid for cid in top_ids)


# ─────────────────────────────────────────────────────────────────────
# Latency tests (< 200ms target)
# ─────────────────────────────────────────────────────────────────────

class TestRetrievalLatency:
    """Assert that retrieval completes within 200ms for various queries."""

    QUERIES = [
        ("What is the capital of India?", "eng"),
        ("speed of light", "eng"),
        ("Python programming language", "eng"),
        ("भारत की राजधानी", "hin"),
        ("प्रकाश की गति", "hin"),
        ("Great Wall of China", "eng"),
        ("machine learning artificial intelligence", "eng"),
        ("financial capital Mumbai", "eng"),
        ("Einstein relativity", "eng"),
        ("Guido van Rossum", "eng"),
    ]

    def test_latency_under_200ms(self, corpus_chunks):
        retriever = HybridRetriever(corpus_chunks, fusion="rrf")

        # Warm-up query (first query may include model loading)
        retriever.retrieve("warm up query", k=3)

        latencies = []
        for query, lang in self.QUERIES:
            _, latency_ms = retriever.retrieve_with_latency(query, k=5, lang=lang)
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]

        print(f"\n--- Retrieval Latency Results ---")
        print(f"Avg: {avg_latency:.1f}ms | P50: {p50:.1f}ms | Max: {max(latencies):.1f}ms")

        # P50 should be under 200ms for this small corpus
        assert p50 < 200, f"P50 latency {p50:.1f}ms exceeds 200ms target"
