"""
Retrieval Package initialization.
Exposes top-level functions for retrieval.
"""

from retrieval.hybrid import HybridRetriever
from chunking.types import Chunk
from typing import List, Optional, Tuple

_GLOBAL_PIPELINE: Optional[HybridRetriever] = None

def build_retrieval_pipeline(chunks: List[Chunk]) -> HybridRetriever:
    """Build and index in-process hybrid retriever."""
    global _GLOBAL_PIPELINE
    pipeline = HybridRetriever()
    pipeline.index(chunks)
    _GLOBAL_PIPELINE = pipeline
    return pipeline

def retrieve(
    query: str,
    k: int = 5,
    language_filter: Optional[str] = None,
    query_cluster_filter: Optional[str] = None,
    strategy: str = "hybrid"
) -> List[Chunk]:
    """Single retrieval entrypoint."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        raise RuntimeError("Retrieval pipeline not initialized. Call build_retrieval_pipeline(chunks) first.")

    results_with_scores = _GLOBAL_PIPELINE.search(
        query=query,
        k=k,
        language_filter=language_filter,
        query_cluster_filter=query_cluster_filter,
        strategy=strategy
    )
    return [chunk for chunk, score in results_with_scores]

def retrieve_with_scores(
    query: str,
    k: int = 5,
    language_filter: Optional[str] = None,
    query_cluster_filter: Optional[str] = None,
    strategy: str = "hybrid"
) -> List[Tuple[Chunk, float]]:
    """Retrieval entrypoint returning chunks with relevance scores."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        raise RuntimeError("Retrieval pipeline not initialized. Call build_retrieval_pipeline(chunks) first.")

    return _GLOBAL_PIPELINE.search(
        query=query,
        k=k,
        language_filter=language_filter,
        query_cluster_filter=query_cluster_filter,
        strategy=strategy
    )
