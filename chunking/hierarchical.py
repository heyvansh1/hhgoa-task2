from typing import List
from data.preprocess import Passage
from chunking.types import Chunk
from chunking.interface import Chunker
from chunking.fixed_size import FixedSizeChunker

class HierarchicalChunker(Chunker):
    """
    Hierarchical (Parent-Child) Chunker:
    Splits text into small child retrieval chunks (~40 tokens) linked directly
    to larger parent passage contexts via parent_chunk_id and parent_text.
    """
    def __init__(self, child_chunk_size: int = 40, child_overlap_percent: float = 0.20):
        self.child_chunker = FixedSizeChunker(chunk_size=child_chunk_size, overlap_percent=child_overlap_percent)

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        hierarchical_chunks: List[Chunk] = []

        for passage in passages:
            parent_id = f"parent_{passage.passage_id}"
            parent_text = passage.text

            # Generate child chunks from the parent passage
            child_chunks = self.child_chunker.chunk([passage])

            for idx, child in enumerate(child_chunks):
                hierarchical_chunks.append(
                    Chunk(
                        chunk_id=f"child_{passage.passage_id}_{idx}",
                        source_passage_id=passage.passage_id,
                        language=passage.language,
                        text=child.text,
                        query_cluster=passage.query_cluster,
                        parent_chunk_id=parent_id,
                        parent_text=parent_text,
                        metadata={
                            "strategy": "hierarchical",
                            "child_index": idx,
                            "parent_length_words": len(parent_text.split())
                        }
                    )
                )

        return hierarchical_chunks
