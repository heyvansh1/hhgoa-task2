"""Phase 4: Chunking Strategy Benchmark.

Evaluates all four chunking strategies (Passage-Native, Fixed-Size, Semantic,
Hierarchical) on 50 English + 50 Hindi queries.  Computes Recall@1, Recall@3,
Recall@5, MRR@5, and average retrieval latency.

Generates:
    - ``eval/chunking_benchmark_results.md``  — comparison table
    - ``eval/chunking_comparison.png``         — grouped bar chart
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import Passage, normalize_text
from chunking.types import Chunk
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SemanticChunker
from chunking.hierarchical import HierarchicalChunker
from retrieval.retrieval import HybridRetriever

logger = logging.getLogger(__name__)

from data.corpus import _EN_QA_PAIRS, _HI_QA_PAIRS, _DISTRACTOR_PASSAGES


# ---------------------------------------------------------------------------
# Build corpus and queries
# ---------------------------------------------------------------------------

def _build_corpus() -> Tuple[List[Passage], List[dict]]:
    """Build the benchmark corpus and query set.

    Returns:
        (passages, queries) where each query is a dict with keys:
        ``query``, ``relevant_passage_id``, ``language``.
    """
    passages: List[Passage] = []
    queries: List[dict] = []

    for query, text, pid in _EN_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "eng", pid))
        queries.append({"query": query, "relevant_passage_id": pid, "language": "eng"})

    for query, text, pid in _HI_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "hin", pid))
        queries.append({"query": query, "relevant_passage_id": pid, "language": "hin"})

    for text, pid, lang in _DISTRACTOR_PASSAGES:
        passages.append(Passage(pid, normalize_text(text), lang, pid))

    return passages, queries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _recall_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def _mrr_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid == relevant_id:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    recall_1: float = 0.0
    recall_3: float = 0.0
    recall_5: float = 0.0
    mrr_5: float = 0.0
    avg_latency_ms: float = 0.0
    # per-language breakdowns
    recall_5_en: float = 0.0
    recall_5_hi: float = 0.0


def run_benchmark() -> List[StrategyResult]:
    """Run the full benchmark and return results for each strategy."""
    passages, queries = _build_corpus()

    strategies = {
        "Passage-Native": PassageNativeChunker(),
        "Fixed-Size (256/20%)": FixedSizeChunker(chunk_size=256, overlap_percent=0.20),
        "Semantic": SemanticChunker(threshold_percentile=75.0),
        "Hierarchical (128/25%)": HierarchicalChunker(child_token_size=128, child_overlap_percent=0.25),
    }

    results: List[StrategyResult] = []

    for name, chunker in strategies.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        # 1. Chunk the corpus
        chunks = chunker.chunk(passages)
        # For hierarchical, only use child chunks for retrieval
        if name.startswith("Hierarchical"):
            retrieval_chunks = [c for c in chunks if c.metadata.get("hierarchy_level") != "parent"]
        else:
            retrieval_chunks = chunks

        print(f"  Chunks produced: {len(retrieval_chunks)}")

        # 2. Build retriever
        retriever = HybridRetriever(retrieval_chunks, fusion="rrf")

        # 3. Warm-up
        retriever.retrieve("warm up query", k=5)

        # 4. Evaluate
        r1s, r3s, r5s, mrrs, latencies = [], [], [], [], []
        r5_en, r5_hi = [], []

        for q in queries:
            retrieved, lat = retriever.retrieve_with_latency(
                q["query"], k=5, lang=q["language"]
            )
            retrieved_passage_ids = [c.source_passage_id for c in retrieved]
            relevant = q["relevant_passage_id"]

            r1s.append(_recall_at_k(retrieved_passage_ids, relevant, 1))
            r3s.append(_recall_at_k(retrieved_passage_ids, relevant, 3))
            r5s.append(_recall_at_k(retrieved_passage_ids, relevant, 5))
            mrrs.append(_mrr_at_k(retrieved_passage_ids, relevant, 5))
            latencies.append(lat)

            if q["language"] == "eng":
                r5_en.append(_recall_at_k(retrieved_passage_ids, relevant, 5))
            else:
                r5_hi.append(_recall_at_k(retrieved_passage_ids, relevant, 5))

        sr = StrategyResult(
            name=name,
            recall_1=np.mean(r1s),
            recall_3=np.mean(r3s),
            recall_5=np.mean(r5s),
            mrr_5=np.mean(mrrs),
            avg_latency_ms=np.mean(latencies),
            recall_5_en=np.mean(r5_en) if r5_en else 0.0,
            recall_5_hi=np.mean(r5_hi) if r5_hi else 0.0,
        )
        results.append(sr)
        print(f"  Recall@1={sr.recall_1:.3f}  Recall@3={sr.recall_3:.3f}  "
              f"Recall@5={sr.recall_5:.3f}  MRR@5={sr.mrr_5:.3f}  "
              f"Latency={sr.avg_latency_ms:.1f}ms")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(results: List[StrategyResult], output_path: str) -> None:
    """Write the benchmark comparison to a markdown file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Chunking Strategy Benchmark Results\n\n")
        f.write("Evaluated on **50 English + 50 Hindi** queries with hybrid retrieval (RRF fusion).\n\n")

        # Main table
        f.write("## Overall Comparison\n\n")
        f.write("| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Avg Latency (ms) |\n")
        f.write("|----------|----------|----------|----------|-------|-------------------|\n")
        for r in results:
            f.write(f"| {r.name} | {r.recall_1:.3f} | {r.recall_3:.3f} | {r.recall_5:.3f} | {r.mrr_5:.3f} | {r.avg_latency_ms:.1f} |\n")

        # Per-language breakdown
        f.write("\n## Per-Language Recall@5\n\n")
        f.write("| Strategy | English R@5 | Hindi R@5 |\n")
        f.write("|----------|-------------|------------|\n")
        for r in results:
            f.write(f"| {r.name} | {r.recall_5_en:.3f} | {r.recall_5_hi:.3f} |\n")

        # Key findings
        best = max(results, key=lambda x: x.recall_5)
        fastest = min(results, key=lambda x: x.avg_latency_ms)
        f.write("\n## Key Findings\n\n")
        f.write(f"- **Best Recall@5**: {best.name} ({best.recall_5:.3f})\n")
        f.write(f"- **Lowest Latency**: {fastest.name} ({fastest.avg_latency_ms:.1f}ms)\n")
        f.write(f"- **Best MRR@5**: {max(results, key=lambda x: x.mrr_5).name} "
                f"({max(results, key=lambda x: x.mrr_5).mrr_5:.3f})\n")

        f.write("\n## Trade-offs\n\n")
        f.write("- **Passage-Native**: Zero overhead, preserves original boundaries. Best when passages are already well-scoped.\n")
        f.write("- **Fixed-Size**: Predictable chunk sizes. Works well for uniform-length documents but splits mid-sentence.\n")
        f.write("- **Semantic**: Respects topic boundaries. Higher quality splits but requires embedding computation.\n")
        f.write("- **Hierarchical**: Parent-child structure enables context expansion. Highest chunk count but best for complex queries.\n")

        f.write(f"\n*Generated automatically by `eval/benchmark_chunking.py`*\n")

    print(f"\nReport written to: {output_path}")


def _generate_bar_chart(results: List[StrategyResult], output_path: str) -> None:
    """Generate a grouped bar chart comparing strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    metrics = {
        "Recall@1": [r.recall_1 for r in results],
        "Recall@3": [r.recall_3 for r in results],
        "Recall@5": [r.recall_5 for r in results],
        "MRR@5": [r.mrr_5 for r in results],
    }

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Chunking Strategy", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Chunking Strategy Benchmark — Retrieval Quality Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full benchmark and generate reports."""
    results = run_benchmark()

    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)

    _generate_markdown_report(results, os.path.join(output_dir, "chunking_benchmark_results.md"))
    _generate_bar_chart(results, os.path.join(output_dir, "chunking_comparison.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
