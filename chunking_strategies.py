"""
Chunking Strategies Unified Entrypoint (Phase 2 Deliverable)
Provides a shared interface `chunk(passages) -> List[Chunk]` across all chunking strategies.
"""

from typing import List, Dict, Type
from data.preprocess import Passage
from chunking.types import Chunk
from chunking.interface import Chunker
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SemanticChunker
from chunking.metadata_aware import MetadataAwareChunker
from chunking.hierarchical import HierarchicalChunker

STRATEGIES: Dict[str, Type[Chunker]] = {
    "passage_native": PassageNativeChunker,
    "fixed_size": FixedSizeChunker,
    "semantic": SemanticChunker,
    "metadata_aware": MetadataAwareChunker,
    "hierarchical": HierarchicalChunker,
}

def get_chunker(strategy_name: str, **kwargs) -> Chunker:
    """Factory function to get a chunker instance by strategy name."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy_name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy_name](**kwargs)

def chunk_with_strategy(passages: List[Passage], strategy_name: str, **kwargs) -> List[Chunk]:
    """Helper function to chunk passages using a named strategy."""
    chunker = get_chunker(strategy_name, **kwargs)
    return chunker.chunk(passages)

if __name__ == "__main__":
    from data.preprocess import load_msmarco_subset
    
    print("Testing Phase 2 Chunking Strategies...")
    passages = load_msmarco_subset(num_queries=2)
    
    for name in STRATEGIES.keys():
        chk = get_chunker(name)
        chunks = chk.chunk(passages)
        print(f"Strategy [{name:15s}]: produced {len(chunks)} chunks from {len(passages)} passages.")
