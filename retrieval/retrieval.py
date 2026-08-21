"""Hybrid retrieval engine: Dense (FAISS) + Sparse (BM25) with fusion.

Provides sub-200ms in-process retrieval over multilingual chunks using:
  - **Dense path**: Multilingual sentence embeddings indexed in FAISS (IVFFlat/FlatIP).
  - **Sparse path**: BM25 via rank_bm25 for exact keyword matching.
  - **Fusion**: Reciprocal Rank Fusion (RRF) or Weighted Score Fusion.

Public API::

    retriever = HybridRetriever(chunks)
    results = retriever.retrieve(query="What is the capital of India?", k=5, lang="eng")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from chunking.types import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model singleton (shared with chunking/semantic.py)
# ---------------------------------------------------------------------------
_EMBEDDER = None
_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_embedder(model_name: str = _DEFAULT_MODEL):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", model_name)
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Scored result
# ---------------------------------------------------------------------------


@dataclass
class ScoredChunk:
    """A chunk with an associated relevance score."""

    chunk: Chunk
    score: float
    source: str = "hybrid"  # "dense", "sparse", or "hybrid"


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------


class FusionMethod(str, Enum):
    RRF = "rrf"
    WEIGHTED = "weighted"


def _reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredChunk]],
    k_rrf: int = 60,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion (RRF).

    For each document, RRF score = Σ 1 / (k + rank_i) across all lists.

    Args:
        ranked_lists: Each inner list is sorted by descending relevance.
        k_rrf: Smoothing constant (default 60 per the original paper).

    Returns:
        Mapping from chunk_id → fused score.
    """
    scores: Dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, sc in enumerate(rlist):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
    return scores


def _weighted_score_fusion(
    ranked_lists: List[List[ScoredChunk]],
    weights: List[float],
) -> Dict[str, float]:
    """Weighted linear combination of normalised scores.

    Each list's scores are min-max normalised to [0, 1], then weighted.

    Args:
        ranked_lists: Each inner list is sorted by descending relevance.
        weights: One weight per ranked list. Must sum to 1.

    Returns:
        Mapping from chunk_id → fused score.
    """
    scores: Dict[str, float] = {}
    for rlist, w in zip(ranked_lists, weights):
        if not rlist:
            continue
        raw = np.array([sc.score for sc in rlist])
        mn, mx = raw.min(), raw.max()
        normed = (raw - mn) / (mx - mn + 1e-9)
        for sc, n in zip(rlist, normed):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + float(n) * w
    return scores


# ---------------------------------------------------------------------------
# Dense retriever (FAISS)
# ---------------------------------------------------------------------------


class DenseRetriever:
    """In-process dense retrieval using FAISS flat index.

    Embeds all chunks at build time, then answers queries via exact
    inner-product search in embedding space.

    Args:
        model_name: SentenceTransformer model for embedding.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.model_name = model_name
        self._index = None
        self._id_map: Dict[int, str] = {}  # faiss internal id → chunk_id
        self._dim: int = 0
        self._num_items: int = 0

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Embed *chunks* and build the FAISS index."""
        import faiss

        embedder = _get_embedder(self.model_name)
        texts = [c.text for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=256)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self._dim = embeddings.shape[1]
        self._num_items = len(chunks)

        # Use flat inner product index (exact search, fast for small corpora)
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)

        self._index = index
        self._id_map = {i: c.chunk_id for i, c in enumerate(chunks)}

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[ScoredChunk]:
        """Return top-*k* nearest chunks for *query_embedding*."""
        if self._index is None:
            return []

        import faiss

        q = query_embedding.copy().astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)

        actual_k = min(k, self._num_items)
        scores, indices = self._index.search(q, actual_k)

        results: List[ScoredChunk] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            cid = self._id_map[int(idx)]
            results.append(
                ScoredChunk(
                    chunk=Chunk(chunk_id=cid, text="", source_passage_id="", language=""),
                    score=float(score),
                    source="dense",
                )
            )
        return results


# ---------------------------------------------------------------------------
# Sparse retriever (BM25)
# ---------------------------------------------------------------------------


class SparseRetriever:
    """In-memory BM25 retrieval using rank_bm25.

    Tokenises chunks via whitespace splitting, which works adequately for
    both English and Hindi (Devanagari words are space-separated).
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Index *chunks* for BM25 retrieval."""
        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        """Return top-*k* BM25-scored chunks for *query*."""
        if self._bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k_idx = np.argsort(scores)[::-1][:k]
        results: List[ScoredChunk] = []
        for idx in top_k_idx:
            if scores[idx] <= 0:
                continue
            results.append(
                ScoredChunk(
                    chunk=Chunk(chunk_id=self._chunk_ids[idx], text="", source_passage_id="", language=""),
                    score=float(scores[idx]),
                    source="sparse",
                )
            )
        return results


# ---------------------------------------------------------------------------
# Hybrid retriever (Dense + Sparse + Fusion)
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combined dense + sparse retrieval with configurable fusion.

    Supports metadata pre-filtering by language and configurable fusion
    via RRF or Weighted Score Fusion.

    Args:
        chunks: The corpus of chunks to index.
        fusion: Fusion method — ``"rrf"`` or ``"weighted"``.
        dense_weight: Weight for dense scores in weighted fusion (sparse
            weight = 1 - dense_weight).
        model_name: Embedding model name.

    Example::

        retriever = HybridRetriever(my_chunks, fusion="rrf")
        results = retriever.retrieve("What is the capital of India?", k=5, lang="eng")
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        fusion: str = "rrf",
        dense_weight: float = 0.6,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.fusion = FusionMethod(fusion)
        self.dense_weight = dense_weight
        self.model_name = model_name

        # Store chunks indexed by ID for full object look-up after fusion
        self._chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

        # Language-partitioned indexes for pre-filtering
        self._lang_chunks: Dict[str, List[Chunk]] = {}
        for c in chunks:
            self._lang_chunks.setdefault(c.language, []).append(c)

        # Build sub-retrievers for every language partition + an "all" index
        self._dense_indexes: Dict[str, DenseRetriever] = {}
        self._sparse_indexes: Dict[str, SparseRetriever] = {}

        for lang, lang_chunks in self._lang_chunks.items():
            dr = DenseRetriever(model_name=model_name)
            dr.build(lang_chunks)
            self._dense_indexes[lang] = dr

            sr = SparseRetriever()
            sr.build(lang_chunks)
            self._sparse_indexes[lang] = sr

        # Also build an "all" index
        all_chunks = list(chunks)
        dr_all = DenseRetriever(model_name=model_name)
        dr_all.build(all_chunks)
        self._dense_indexes["all"] = dr_all

        sr_all = SparseRetriever()
        sr_all.build(all_chunks)
        self._sparse_indexes["all"] = sr_all

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        lang: Optional[str] = None,
    ) -> List[Chunk]:
        """Retrieve the top-*k* chunks for *query* using hybrid fusion.

        Args:
            query: The user's natural-language query string.
            k: Number of results to return.
            lang: Optional language filter (e.g. ``"eng"``, ``"hin"``).
                  If ``None``, searches the full corpus.

        Returns:
            A list of :class:`Chunk` objects ranked by fused relevance,
            with full text and metadata populated.
        """
        index_key = lang if (lang and lang in self._dense_indexes) else "all"

        embedder = _get_embedder(self.model_name)
        query_emb = embedder.encode([query], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Dense search
        dense_results = self._dense_indexes[index_key].search(query_emb[0], k=k * 2)
        # Sparse search
        sparse_results = self._sparse_indexes[index_key].search(query, k=k * 2)

        # Fuse
        if self.fusion == FusionMethod.RRF:
            fused_scores = _reciprocal_rank_fusion([dense_results, sparse_results])
        else:
            fused_scores = _weighted_score_fusion(
                [dense_results, sparse_results],
                [self.dense_weight, 1.0 - self.dense_weight],
            )

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]

        # Resolve to full Chunk objects
        return [self._chunk_map[cid] for cid in sorted_ids if cid in self._chunk_map]

    def retrieve_with_latency(
        self,
        query: str,
        k: int = 5,
        lang: Optional[str] = None,
    ) -> tuple:
        """Like :meth:`retrieve`, but also returns latency in milliseconds.

        Returns:
            ``(chunks, latency_ms)``
        """
        start = time.perf_counter()
        chunks = self.retrieve(query, k=k, lang=lang)
        latency_ms = (time.perf_counter() - start) * 1000
        return chunks, latency_ms
