"""Passage-native chunking strategy (Baseline #1).

Each existing dataset passage is preserved as exactly one retrieval unit,
maintaining the original passage boundaries from MSMARCO-XI.
"""

from typing import List

from data.preprocess import Passage
from chunking.interface import BaseChunker
from chunking.types import Chunk


class PassageNativeChunker(BaseChunker):
    """Baseline #1: Passage-native chunking.

    Each existing dataset passage remains exactly one retrieval unit.
    No splitting or merging is performed.
    """

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Convert passages 1:1 into chunks.

        Args:
            passages: Source passages from the dataset.

        Returns:
            One Chunk per Passage, with metadata populated.
        """
        chunks: List[Chunk] = []
        for p in passages:
            chunks.append(
                Chunk(
                    chunk_id=f"pn_{p.passage_id}_0",
                    text=p.text,
                    source_passage_id=p.passage_id,
                    language=p.language,
                    metadata={
                        "strategy": "passage_native",
                        "char_length": len(p.text),
                        "token_length": len(p.text.split()),
                    },
                    query_cluster=p.query_cluster,
                )
            )
        return chunks
