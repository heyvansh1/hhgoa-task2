"""Lightweight TF-IDF retrieval engine for Render free-tier deployment.

Replaces sentence-transformers + FAISS (~400MB RAM) with pure-Python
TF-IDF + cosine similarity (~0.5MB RAM).  Uses only stdlib + math.

Public API mirrors HybridRetriever::

    retriever = LiteRetriever()
    results = retriever.retrieve("What is the capital of India?", k=5, lang="eng")
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ─── Data types (mirroring chunking.types.Chunk) ─────────────────────

@dataclass
class Chunk:
    """Minimal chunk type compatible with the pipeline."""
    chunk_id: str
    text: str
    source_passage_id: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_cluster: Optional[str] = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


# ─── TF-IDF Engine ───────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokeniser (works for EN + HI)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [t for t in re.split(r"\s+", text) if len(t) > 1]


class TFIDFIndex:
    """In-memory TF-IDF index with cosine similarity search."""

    def __init__(self) -> None:
        self._docs: List[Chunk] = []
        self._tfidf_vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._norms: List[float] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._docs = list(chunks)
        n = len(chunks)
        if n == 0:
            return

        # Document frequency
        df: Dict[str, int] = defaultdict(int)
        doc_tokens: List[List[str]] = []
        for c in chunks:
            tokens = _tokenize(c.text)
            doc_tokens.append(tokens)
            for term in set(tokens):
                df[term] += 1

        # IDF: log(N / df)
        self._idf = {term: math.log(n / freq) for term, freq in df.items()}

        # TF-IDF vectors + norms
        self._tfidf_vectors = []
        self._norms = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                tfidf = (count / total) * self._idf.get(term, 0.0)
                if tfidf > 0:
                    vec[term] = tfidf
                    norm_sq += tfidf * tfidf
            self._tfidf_vectors.append(vec)
            self._norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1e-9)

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        # Query TF-IDF
        tf = Counter(tokens)
        total = len(tokens)
        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0
        for term, count in tf.items():
            tfidf = (count / total) * self._idf.get(term, 0.0)
            if tfidf > 0:
                q_vec[term] = tfidf
                q_norm_sq += tfidf * tfidf
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1e-9

        # Cosine similarity
        scores: List[tuple] = []
        for i, doc_vec in enumerate(self._tfidf_vectors):
            dot = sum(q_vec.get(t, 0.0) * doc_vec.get(t, 0.0) for t in q_vec)
            sim = dot / (q_norm * self._norms[i])
            if sim > 0:
                scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=self._docs[i], score=s)
            for i, s in scores[:k]
        ]


# ─── Corpus builder ──────────────────────────────────────────────────

def _build_corpus() -> List[Chunk]:
    """Build corpus from the benchmark QA pairs (same data as main.py)."""
    from data.corpus import _EN_QA_PAIRS, _HI_QA_PAIRS, _DISTRACTOR_PASSAGES

    chunks: List[Chunk] = []

    for query, text, pid in _EN_QA_PAIRS:
        text_clean = unicodedata.normalize("NFKC", text)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()
        chunks.append(Chunk(
            chunk_id=f"pn_{pid}_0",
            text=text_clean,
            source_passage_id=pid,
            language="eng",
        ))

    for query, text, pid in _HI_QA_PAIRS:
        text_clean = unicodedata.normalize("NFKC", text)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()
        chunks.append(Chunk(
            chunk_id=f"pn_{pid}_0",
            text=text_clean,
            source_passage_id=pid,
            language="hin",
        ))

    for text, pid, lang in _DISTRACTOR_PASSAGES:
        text_clean = unicodedata.normalize("NFKC", text)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()
        chunks.append(Chunk(
            chunk_id=f"pn_{pid}_0",
            text=text_clean,
            source_passage_id=pid,
            language=lang,
        ))

    return chunks


# ─── Public retriever class ──────────────────────────────────────────

class LiteRetriever:
    """Lightweight TF-IDF retriever with language partitioning.

    Drop-in replacement for HybridRetriever, using ~0.5MB RAM
    instead of ~400MB.
    """

    def __init__(self) -> None:
        self._chunk_map: Dict[str, Chunk] = {}
        self._indexes: Dict[str, TFIDFIndex] = {}
        self._all_chunks: List[Chunk] = []

        chunks = _build_corpus()
        self._all_chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}

        # Language-partitioned indexes
        lang_chunks: Dict[str, List[Chunk]] = {}
        for c in chunks:
            lang_chunks.setdefault(c.language, []).append(c)

        for lang, lc in lang_chunks.items():
            idx = TFIDFIndex()
            idx.build(lc)
            self._indexes[lang] = idx

        # "all" index
        idx_all = TFIDFIndex()
        idx_all.build(chunks)
        self._indexes["all"] = idx_all

    def retrieve(self, query: str, k: int = 5, lang: Optional[str] = None) -> List[Chunk]:
        index_key = lang if (lang and lang in self._indexes) else "all"
        results = self._indexes[index_key].search(query, k=k)
        return [sc.chunk for sc in results]

    def retrieve_with_latency(self, query: str, k: int = 5, lang: Optional[str] = None) -> tuple:
        start = time.perf_counter()
        chunks = self.retrieve(query, k=k, lang=lang)
        latency_ms = (time.perf_counter() - start) * 1000
        return chunks, latency_ms
