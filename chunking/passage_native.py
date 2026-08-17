from typing import List
from data.preprocess import Passage
from chunking.types import Chunk

class PassageNativeChunker:
    """
    Baseline #1: Passage-native chunking.
    Each existing dataset passage remains exactly one retrieval unit.
    """
    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        chunks = []
        for i, p in enumerate(passages):
            chunks.append(Chunk(
                chunk_id=f"pn_{p.passage_id}_0", # 0 because it's always 1:1
                source_passage_id=p.passage_id,
                language=p.language,
                text=p.text,
                query_cluster=p.query_cluster
            ))
        return chunks
