"""
Retrieval Module (Phase 3 Deliverable)
Provides single entrypoint `retrieve(query, k) -> List[Chunk]`, hybrid by default.
Ensures retrieval latency is under 200ms.
"""

import time
from typing import List, Optional, Tuple
from chunking.types import Chunk
from retrieval.hybrid import HybridRetriever

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
    """
    Single retrieval entrypoint.
    Returns top-k Chunks matching the query.
    """
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
    """Retrieval entrypoint returning chunks alongside relevance/RRF scores."""
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

if __name__ == "__main__":
    from data.preprocess import load_msmarco_subset
    from chunking_strategies import chunk_with_strategy

    print("Phase 3 Deliverable Test: In-Process Hybrid Retrieval & Latency Check...")
    
    passages = load_msmarco_subset(num_queries=5)
    chunks = chunk_with_strategy(passages, "passage_native")
    
    print(f"Indexing {len(chunks)} chunks...")
    pipeline = build_retrieval_pipeline(chunks)
    
    test_queries = [
        ("What is the capital of India?", "eng"),
        ("भारत की राजधानी क्या है?", "hin"),
        ("Financial capital of India", "eng"),
        ("मुंबई किस देश की राजधानी है?", "hin"),
        ("Speed of light in vacuum", "eng"),
        ("प्रकाश की गति कितनी होती है?", "hin"),
        ("New Delhi Yamuna river", "eng"),
        ("यमुना नदी तट दिल्ली", "hin"),
        ("Universal physical constant c", "eng"),
        ("भौतिकी में नियतांक c", "hin"),
        ("British India capital", "eng"),
        ("ब्रिटिश भारत राजधानी", "hin"),
        ("North-central part of India", "eng"),
        ("299792458 metres per second", "eng"),
        ("300000 km/s speed", "eng"),
        ("Mumbai financial", "eng"),
        ("निर्वात में प्रकाश", "hin"),
        ("यमुना नदी", "hin"),
        ("Speed of light value", "eng"),
        ("India capital Delhi", "eng"),
    ]

    latencies_ms = []
    print("\n--- Running 20 Query Latency Benchmark ---")
    for q, lang in test_queries:
        start_time = time.perf_counter()
        results = retrieve(q, k=3, language_filter=lang)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed_ms)
        safe_q = q.encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"Query: [{safe_q:35s}] | Lang Filter: {lang} | Latency: {elapsed_ms:6.2f} ms | Results: {len(results)}")

    latencies_ms.sort()
    p50_latency = latencies_ms[len(latencies_ms) // 2]
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    
    print(f"\n--- Benchmark Summary ---")
    print(f"P50 Latency : {p50_latency:.2f} ms")
    print(f"Avg Latency : {avg_latency:.2f} ms")
    print(f"Budget Check (<200ms) : {'PASSED' if p50_latency < 200.0 else 'FAILED'}")
