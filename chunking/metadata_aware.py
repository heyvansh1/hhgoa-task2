from typing import List, Optional
from data.preprocess import Passage
from chunking.types import Chunk
from chunking.interface import Chunker
from chunking.fixed_size import FixedSizeChunker

class MetadataAwareChunker(Chunker):
    """
    Metadata-Aware Chunker:
    Chunks passages while explicitly attaching structured metadata headers 
    and dictionary tags for pre-filtering (e.g. language, query_cluster, source_passage).
    """
    def __init__(self, base_chunker: Optional[Chunker] = None, prepend_header: bool = True):
        self.base_chunker = base_chunker or FixedSizeChunker(chunk_size=100, overlap_percent=0.20)
        self.prepend_header = prepend_header

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        base_chunks = self.base_chunker.chunk(passages)
        metadata_chunks: List[Chunk] = []

        for c in base_chunks:
            meta = {
                "strategy": "metadata_aware",
                "source_passage_id": c.source_passage_id,
                "language": c.language,
                "query_cluster": c.query_cluster
            }
            
            header = f"[Lang: {c.language} | Cluster: {c.query_cluster or 'N/A'} | Source: {c.source_passage_id}] "
            chunk_text = (header + c.text) if self.prepend_header else c.text

            metadata_chunks.append(
                Chunk(
                    chunk_id=f"meta_{c.chunk_id}",
                    source_passage_id=c.source_passage_id,
                    language=c.language,
                    text=chunk_text,
                    query_cluster=c.query_cluster,
                    metadata=meta
                )
            )

        return metadata_chunks
