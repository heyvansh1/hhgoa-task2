"""Fixed-size + overlap chunking strategy (Baseline #2).

Uses a simple whitespace tokenizer (words ≈ tokens) to split passages into
fixed-size windows with configurable overlap.  Default: 256 tokens, 20% overlap.
"""

from typing import List

from data.preprocess import Passage
from chunking.interface import BaseChunker
from chunking.types import Chunk


class FixedSizeChunker(BaseChunker):
    """Baseline #2: Fixed-size + overlap chunking.

    Uses a simple whitespace tokenizer (words = tokens) to keep dependencies
    lightweight.  Target size: 256 tokens, Overlap: 20%.

    Args:
        chunk_size: Number of whitespace tokens per chunk window.
        overlap_percent: Fraction of ``chunk_size`` used as overlap between
            consecutive windows (0.0–1.0).
    """

    def __init__(self, chunk_size: int = 256, overlap_percent: float = 0.20) -> None:
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_percent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer suitable for English and Hindi text."""
        return text.split()

    @staticmethod
    def _detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Split passages into fixed-size token windows with overlap.

        Args:
            passages: Source passages from the dataset.

        Returns:
            A list of Chunks, each containing ``chunk_size`` tokens (except
            possibly the last chunk of each passage).
        """
        chunks: List[Chunk] = []

        for p in passages:
            tokens = self._tokenize(p.text)
            if not tokens:
                continue

            step = max(self.chunk_size - self.overlap, 1)
            chunk_idx = 0

            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + self.chunk_size]
                chunk_text = self._detokenize(chunk_tokens)

                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=f"fs_{p.passage_id}_{chunk_idx}",
                            text=chunk_text,
                            source_passage_id=p.passage_id,
                            language=p.language,
                            metadata={
                                "strategy": "fixed_size",
                                "chunk_index": chunk_idx,
                                "token_length": len(chunk_tokens),
                                "char_length": len(chunk_text),
                                "chunk_size_cfg": self.chunk_size,
                                "overlap_cfg": self.overlap,
                            },
                            query_cluster=p.query_cluster,
                        )
                    )
                    chunk_idx += 1

                # Break if this chunk reached the end of the text
                if i + self.chunk_size >= len(tokens):
                    break

        return chunks
