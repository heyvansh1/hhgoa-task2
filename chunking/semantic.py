"""Semantic chunking strategy.

Uses a lightweight multilingual sentence embedder to compute cosine distances
between adjacent sentences.  Chunk boundaries are inserted at distance
*spikes* — i.e. where the semantic topic shifts — giving retrieval units that
are topically coherent.

Model: ``paraphrase-multilingual-MiniLM-L12-v2`` (~134 M params, supports
100+ languages including English and Hindi).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from data.preprocess import Passage
from chunking.interface import BaseChunker
from chunking.types import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model singleton — avoid re-loading the ~420 MB model every time
# ---------------------------------------------------------------------------
_EMBEDDER: Optional[object] = None
_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_embedder(model_name: str = _DEFAULT_MODEL):
    """Return a cached ``SentenceTransformer`` instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", model_name)
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Sentence splitting (language-aware)
# ---------------------------------------------------------------------------

# Hindi sentence-ending punctuation: purna viram (।), double danda (॥),
# plus standard ASCII period / question / exclamation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using punctuation heuristics.

    Handles both English (period / ? / !) and Hindi (purna viram / danda)
    sentence terminators.

    Returns:
        A list of non-empty sentence strings.
    """
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """Semantic chunker using multilingual sentence embeddings.

    Algorithm:
        1. Split each passage into sentences.
        2. Embed every sentence with ``paraphrase-multilingual-MiniLM-L12-v2``.
        3. Compute cosine distance between consecutive sentence embeddings.
        4. Identify *breakpoints* where the distance exceeds the
           ``threshold_percentile`` of all distances (i.e. the biggest
           topic shifts).
        5. Group sentences between breakpoints into chunks.

    Args:
        model_name: HuggingFace model identifier for the sentence embedder.
        threshold_percentile: Percentile (0–100) of inter-sentence distances
            above which a split is triggered.  Higher values → fewer, larger
            chunks.
        min_chunk_sentences: Minimum number of sentences per chunk.  Short
            trailing segments are merged into the previous chunk.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        threshold_percentile: float = 75.0,
        min_chunk_sentences: int = 1,
    ) -> None:
        self.model_name = model_name
        self.threshold_percentile = threshold_percentile
        self.min_chunk_sentences = min_chunk_sentences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine distances between consecutive embedding vectors.

        Returns an array of length ``len(embeddings) - 1``.
        """
        # Normalise rows to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        normed = embeddings / norms
        # Cosine similarity between adjacent rows
        similarities = np.sum(normed[:-1] * normed[1:], axis=1)
        return 1.0 - similarities  # convert to distance

    def _find_breakpoints(self, distances: np.ndarray) -> List[int]:
        """Return indices where the semantic distance exceeds the threshold.

        Each index ``i`` in the returned list means a split should be
        inserted *after* sentence ``i``.
        """
        if len(distances) == 0:
            return []
        threshold = float(np.percentile(distances, self.threshold_percentile))
        return [i for i, d in enumerate(distances) if d > threshold]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Split passages into semantically coherent chunks.

        Args:
            passages: Source passages from the dataset.

        Returns:
            A list of Chunks with metadata including the embedding model
            and the distance threshold used.
        """
        embedder = _get_embedder(self.model_name)
        chunks: List[Chunk] = []

        for p in passages:
            sentences = _split_sentences(p.text)
            if not sentences:
                continue

            # Single-sentence passages → one chunk, no embedding needed
            if len(sentences) == 1:
                chunks.append(
                    Chunk(
                        chunk_id=f"sem_{p.passage_id}_0",
                        text=sentences[0],
                        source_passage_id=p.passage_id,
                        language=p.language,
                        metadata={
                            "strategy": "semantic",
                            "embedding_model": self.model_name,
                            "split_threshold_pct": self.threshold_percentile,
                            "num_sentences": 1,
                            "char_length": len(sentences[0]),
                        },
                        query_cluster=p.query_cluster,
                    )
                )
                continue

            # Embed all sentences in batch
            embeddings = embedder.encode(sentences, show_progress_bar=False)
            distances = self._cosine_distances(np.array(embeddings))
            breakpoints = self._find_breakpoints(distances)

            # Build chunk boundaries: list of (start_idx, end_idx) inclusive
            boundaries: List[tuple] = []
            start = 0
            for bp in breakpoints:
                end = bp  # inclusive
                if end - start + 1 >= self.min_chunk_sentences:
                    boundaries.append((start, end))
                    start = end + 1
            # Remaining sentences
            if start < len(sentences):
                boundaries.append((start, len(sentences) - 1))

            # Merge very short trailing chunk into previous if possible
            if (
                len(boundaries) > 1
                and boundaries[-1][1] - boundaries[-1][0] + 1
                < self.min_chunk_sentences
            ):
                prev_start, _ = boundaries[-2]
                _, last_end = boundaries[-1]
                boundaries[-2] = (prev_start, last_end)
                boundaries.pop()

            for chunk_idx, (s, e) in enumerate(boundaries):
                chunk_text = " ".join(sentences[s : e + 1])
                chunks.append(
                    Chunk(
                        chunk_id=f"sem_{p.passage_id}_{chunk_idx}",
                        text=chunk_text,
                        source_passage_id=p.passage_id,
                        language=p.language,
                        metadata={
                            "strategy": "semantic",
                            "embedding_model": self.model_name,
                            "split_threshold_pct": self.threshold_percentile,
                            "num_sentences": e - s + 1,
                            "sentence_range": [s, e],
                            "char_length": len(chunk_text),
                        },
                        query_cluster=p.query_cluster,
                    )
                )

        return chunks
