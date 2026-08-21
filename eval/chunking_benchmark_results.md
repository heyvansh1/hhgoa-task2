# Chunking Strategy Benchmark Results

Evaluated on **50 English + 50 Hindi** queries with hybrid retrieval (RRF fusion).

## Overall Comparison

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Avg Latency (ms) |
|----------|----------|----------|----------|-------|-------------------|
| Passage-Native | 0.950 | 1.000 | 1.000 | 0.975 | 16.2 |
| Fixed-Size (256/20%) | 0.950 | 1.000 | 1.000 | 0.975 | 15.8 |
| Semantic | 0.950 | 1.000 | 1.000 | 0.975 | 16.0 |
| Hierarchical (128/25%) | 0.950 | 1.000 | 1.000 | 0.975 | 15.8 |

## Per-Language Recall@5

| Strategy | English R@5 | Hindi R@5 |
|----------|-------------|------------|
| Passage-Native | 1.000 | 1.000 |
| Fixed-Size (256/20%) | 1.000 | 1.000 |
| Semantic | 1.000 | 1.000 |
| Hierarchical (128/25%) | 1.000 | 1.000 |

## Key Findings

- **Best Recall@5**: Passage-Native (1.000)
- **Lowest Latency**: Hierarchical (128/25%) (15.8ms)
- **Best MRR@5**: Passage-Native (0.975)

## Trade-offs

- **Passage-Native**: Zero overhead, preserves original boundaries. Best when passages are already well-scoped.
- **Fixed-Size**: Predictable chunk sizes. Works well for uniform-length documents but splits mid-sentence.
- **Semantic**: Respects topic boundaries. Higher quality splits but requires embedding computation.
- **Hierarchical**: Parent-child structure enables context expansion. Highest chunk count but best for complex queries.

*Generated automatically by `eval/benchmark_chunking.py`*
