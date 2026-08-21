"""Hierarchical / Metadata-Aware chunking strategy.

Creates a two-level hierarchy of retrieval chunks:

* **Parent chunks** — larger context windows (full passages or large segments)
  used to provide surrounding context after retrieval.
* **Child chunks** — smaller, more focused retrieval units that are actually
  matched against queries during search.

Every chunk is tagged with rich metadata: ``language``, ``char_length``,
``token_length``, ``parent_id``, ``hierarchy_level``, ``child_index``.
This enables metadata pre-filtering (e.g. "only search Hindi chunks") at
retrieval time.
"""

from __future__ import annotations

from typing import List

from data.preprocess import Passage
from chunking.interface import BaseChunker
from chunking.types import Chunk


class HierarchicalChunker(BaseChunker):
    """Metadata-aware hierarchical chunker.

    For each passage, produces:
        * One **parent** chunk containing the full passage text.
        * Multiple **child** chunks of ``child_token_size`` tokens with
          ``child_overlap_percent`` overlap, each linked back to the parent
          via ``metadata["parent_id"]``.

    Args:
        child_token_size: Number of whitespace tokens per child chunk.
        child_overlap_percent: Overlap fraction between consecutive child
            windows (0.0–1.0).
    """

    def __init__(
        self,
        child_token_size: int = 128,
        child_overlap_percent: float = 0.25,
    ) -> None:
        self.child_token_size = child_token_size
        self.child_overlap = int(child_token_size * child_overlap_percent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.split()

    @staticmethod
    def _detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Produce parent + child chunks for every passage.

        Args:
            passages: Source passages from the dataset.

        Returns:
            A flat list containing both parent and child chunks.  Child
            chunks carry ``metadata["parent_id"]`` pointing to the parent's
            ``chunk_id``.
        """
        chunks: List[Chunk] = []

        for p in passages:
            tokens = self._tokenize(p.text)
            if not tokens:
                continue

            parent_id = f"hier_{p.passage_id}_parent"

            # ---- Parent chunk ------------------------------------------------
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text=p.text,
                    source_passage_id=p.passage_id,
                    language=p.language,
                    metadata={
                        "strategy": "hierarchical",
                        "hierarchy_level": "parent",
                        "char_length": len(p.text),
                        "token_length": len(tokens),
                        "language": p.language,
                    },
                    query_cluster=p.query_cluster,
                )
            )

            # ---- Child chunks ------------------------------------------------
            step = max(self.child_token_size - self.child_overlap, 1)

            # If the passage is short enough, produce a single child identical
            # to the parent (avoids empty child lists).
            if len(tokens) <= self.child_token_size:
                chunks.append(
                    Chunk(
                        chunk_id=f"hier_{p.passage_id}_c0",
                        text=p.text,
                        source_passage_id=p.passage_id,
                        language=p.language,
                        metadata={
                            "strategy": "hierarchical",
                            "hierarchy_level": "child",
                            "parent_id": parent_id,
                            "child_index": 0,
                            "char_length": len(p.text),
                            "token_length": len(tokens),
                            "language": p.language,
                        },
                        query_cluster=p.query_cluster,
                    )
                )
                continue

            child_idx = 0
            for i in range(0, len(tokens), step):
                child_tokens = tokens[i : i + self.child_token_size]
                child_text = self._detokenize(child_tokens)

                if child_text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=f"hier_{p.passage_id}_c{child_idx}",
                            text=child_text,
                            source_passage_id=p.passage_id,
                            language=p.language,
                            metadata={
                                "strategy": "hierarchical",
                                "hierarchy_level": "child",
                                "parent_id": parent_id,
                                "child_index": child_idx,
                                "char_length": len(child_text),
                                "token_length": len(child_tokens),
                                "language": p.language,
                            },
                            query_cluster=p.query_cluster,
                        )
                    )
                    child_idx += 1

                if i + self.child_token_size >= len(tokens):
                    break

        return chunks
