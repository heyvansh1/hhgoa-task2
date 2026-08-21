# Latency Analytics Report

Tested on **100 queries** (12 guardrail-triggered).

## Pipeline Latency Percentiles (ms)

| Scope | P50 | P70 | P100 |
|-------|-----|-----|------|
| Full Pipeline | 16.09 | 16.71 | 25.63 |
| Retrieval Core Only | 15.88 | 16.50 | 25.20 |
| Text-to-Text Pipeline | 16.05 | 16.65 | 25.53 |

## Average Per-Stage Latency (ms)

| Stage | Avg Latency (ms) |
|-------|-------------------|
| generation | 0.01 |
| guardrails | 0.13 |
| guardrails_output | 0.21 |
| preprocess | 0.02 |
| retrieval | 16.11 |

## Key Observations

- **Retrieval P50**: 15.88ms ✅ Under 200ms target
- **Full Pipeline P50**: 16.09ms
- **Guardrail triggers**: 12/100 queries

*Report generated from actual pipeline runs.*
