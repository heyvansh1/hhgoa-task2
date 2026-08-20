from typing import List, Tuple, Optional, Dict, Any
from chunking.types import Chunk
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever

class HybridRetriever:
    """
    Hybrid Retriever combining Dense (FAISS) and Sparse (BM25) search 
    using Reciprocal Rank Fusion (RRF) with metadata pre-filtering.
    """
    def __init__(self, dense_weight: float = 0.5, sparse_weight: float = 0.5, rrf_k: int = 60):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.dense_retriever.index(chunks)
        self.sparse_retriever.index(chunks)

    def _filter_candidates(self, language: Optional[str] = None, query_cluster: Optional[str] = None) -> Optional[List[int]]:
        if language is None and query_cluster is None:
            return None

        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            if language is not None and chunk.language != language:
                continue
            if query_cluster is not None and chunk.query_cluster != query_cluster:
                continue
            candidate_indices.append(idx)
        return candidate_indices

    def search(
        self,
        query: str,
        k: int = 5,
        language_filter: Optional[str] = None,
        query_cluster_filter: Optional[str] = None,
        strategy: str = "hybrid"
    ) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        cand_indices = self._filter_candidates(language=language_filter, query_cluster=query_cluster_filter)

        if strategy == "dense":
            return self.dense_retriever.search(query, k=k, candidates_indices=cand_indices)
        elif strategy == "sparse":
            return self.sparse_retriever.search(query, k=k, candidates_indices=cand_indices)

        # Hybrid RRF Search
        top_fetch = min(max(k * 4, 20), len(self.chunks))
        dense_results = self.dense_retriever.search(query, k=top_fetch, candidates_indices=cand_indices)
        sparse_results = self.sparse_retriever.search(query, k=top_fetch, candidates_indices=cand_indices)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Dense RRF scoring
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_map[chunk.chunk_id] = chunk
            rrf_score = self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score

        # Sparse RRF scoring
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_map[chunk.chunk_id] = chunk
            rrf_score = self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in sorted_chunks[:k]]
