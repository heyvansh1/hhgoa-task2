# Voice RAG Pipeline — Project Phase Plan (9-Day Build)

**Goal:** Build a latency-aware, multilingual voice RAG system on MSMARCO-XI, with hybrid retrieval, groundedness guardrails, and full latency analytics — submission-ready with a comparison-driven narrative.

**Scope lock:** English + Hindi only. Sarvam (Saarika) for STT. In-process FAISS/hnswlib for vector search. Hybrid (dense + BM25) retrieval. No blocking reranker.

---

## Phase 0 — Setup & Decision Lock (Day 1)

**Objective:** Lock the two decisions everything else depends on: language scope and STT provider.

- [ ] Set up repo structure: `/stt`, `/chunking`, `/retrieval`, `/harness`, `/guardrails`, `/eval`, `/data`
- [ ] Pull MSMARCO-XI subset for English + Hindi only
- [ ] Get Sarvam and ElevenLabs API access working (smoke test: one call each)
- [ ] Run side-by-side latency + accuracy test: 5–10 sample audio clips (mix of English, Hindi, code-mixed) through both STT APIs
- [ ] Log: latency (ms), transcription accuracy (manual spot-check), cost per call

**Deliverable:** `stt_benchmark.md` — table comparing Sarvam vs ElevenLabs on the 5–10 clips, with a one-line decision justification.

**Exit criteria:** STT provider locked. If Sarvam underperforms unexpectedly, fallback decision documented — don't silently swap later.

---

## Phase 1 — Data Prep & Passage-Native Chunking (Day 2)

**Objective:** Get the baseline chunking strategy working end-to-end before adding complexity.

- [ ] Load MSMARCO-XI passages (English + Hindi), normalize encoding/whitespace
- [ ] Implement **passage-native chunking** (respect existing passage boundaries) — this is baseline #1
- [ ] Implement **fixed-size + overlap chunking** (256 tokens, 20% overlap) — this is baseline #2, the "naive" comparison
- [ ] Store chunk metadata: `chunk_id`, `source_passage_id`, `language`, `query_cluster` (if available)

**Deliverable:** `chunking_baselines.py` + a small sample dump showing both strategies on the same 5 passages side by side.

**Exit criteria:** Both baseline chunkers produce clean, inspectable output.

---

## Phase 2 — Advanced Chunking Strategies (Day 3)

**Objective:** Add the strategies that make the "vast chunking" narrative real — not just one fancy method, but a comparison.

- [ ] Implement **semantic chunking**: sentence-transformer embeddings + cosine-distance breakpoint detection
- [ ] Implement **metadata-aware chunking**: tag chunks with query-cluster/language/passage-id for pre-filtering
- [ ] (Stretch) Implement **hierarchical/parent-child chunking**: small retrieval units linked to larger parent context
- [ ] Pick **2–3 total strategies** to carry forward into evaluation (don't over-scope on Day 3)

**Deliverable:** `chunking_strategies.py` covering all implemented strategies with a shared interface (`chunk(passages) -> List[Chunk]`) so retrieval eval can swap between them.

**Exit criteria:** All chosen strategies run on the full English+Hindi passage set without errors.

---

## Phase 3 — Vector Index & Hybrid Retrieval (Day 4)

**Objective:** Build the retrieval core within the 200ms latency budget.

- [ ] Set up **FAISS or hnswlib in-process** index (no networked vector DB round-trip)
- [ ] Build **dense retrieval** path (embedding model → vector search)
- [ ] Build **sparse retrieval** path (BM25) for exact keyword matches
- [ ] Combine into **hybrid retrieval** (score fusion — e.g. reciprocal rank fusion or weighted sum)
- [ ] Wire metadata filtering (from Phase 2) as a pre-search step

**Deliverable:** `retrieval.py` with a single `retrieve(query, k) -> List[Chunk]` entrypoint, hybrid by default.

**Exit criteria:** Retrieval-only P50 latency measured and under 200ms on a quick 20-query test.

---

## Phase 4 — Chunking Strategy Benchmark (Day 5)

**Objective:** Produce the comparison that judges will look at hardest.

- [ ] Run recall@k (k=1,3,5) for each of the 2–3 chunking strategies from Phase 2, using the retrieval pipeline from Phase 3
- [ ] Run against both baselines (passage-native, fixed-size) as the "before" comparison
- [ ] Build a results table/chart: strategy vs recall@k vs avg chunk retrieval latency

**Deliverable:** `chunking_benchmark_results.md` (+ chart image) — this becomes a core slide/section in the final submission.

**Exit criteria:** Clear winner (or documented trade-off) identified and justified with numbers, not intuition.

---

## Phase 5 — Harness / Pipeline Orchestration (Day 6)

**Objective:** Wire everything into the explicit staged pipeline.

- [ ] Implement pipeline stages as typed functions:
  `AudioInput → STT(retry×2, timeout) → QueryPreprocess → Retrieval(hybrid, timeout) → GroundednessCheck → Generation(structured output, timeout) → PostFilter`
- [ ] Each stage: try/except with a defined fallback (e.g. STT failure → "couldn't hear that, try again")
- [ ] Per-stage latency logging (timestamp in/out per stage, written to a structured log)
- [ ] Wire structured-output generation (constrain to answer + cited chunk IDs)

**Deliverable:** `harness.py` — running end-to-end on one sample audio query, printing stage-by-stage latency.

**Exit criteria:** Full pipeline runs start-to-finish on at least 3 real audio inputs without crashing.

---

## Phase 6 — Guardrails (Day 7)

**Objective:** Implement the "knows when not to answer" requirement — explicitly graded, don't skip.

- [ ] **Input guardrail**: cheap keyword + embedding-similarity-to-corpus check before retrieval runs; short-circuit off-topic queries
- [ ] **Output/groundedness guardrail**: check generated answer's key claims against retrieved chunks
  - Baseline: token/entity overlap check
  - Stretch: NLI-based entailment check
- [ ] Ungrounded or off-topic → return "I don't have enough information" instead of hallucinating
- [ ] Test guardrails against a small set of deliberately off-topic and deliberately unanswerable queries

**Deliverable:** `guardrails.py` + a mini test set (`guardrail_test_cases.md`) showing pass/fail on ~10–15 adversarial queries.

**Exit criteria:** Guardrails correctly reject at least 80% of the adversarial test set.

---

## Phase 7 — Latency Analytics (Day 8)

**Objective:** Produce the honest, transparent latency report.

- [ ] Run **50–100 test queries**: mix of in-domain, off-topic, and edge cases
- [ ] Log per-stage + total latency for every query
- [ ] Compute **P50 / P70 / P100** for:
  - Full pipeline (audio in → answer out)
  - Retrieval-only (to honestly show sub-200ms even if STT/generation aren't)
- [ ] Build latency breakdown chart (stacked bar per stage, or box plot per stage)

**Deliverable:** `latency_report.md` + chart — this is your credibility section; report the real numbers, including where the budget is missed and why.

**Exit criteria:** Report is generated from actual logged runs, not estimated numbers.

---

## Phase 8 — Polish, Docs & Submission Video (Day 9)

**Objective:** Package everything into a coherent, judge-friendly submission.

- [ ] Write final `README.md`: architecture diagram, decisions made (with the Phase 0 STT benchmark and Phase 4 chunking benchmark as evidence), known limitations
- [ ] Clean up code, remove dead experiments, pin dependencies
- [ ] Record demo video: show a live query end-to-end, show the chunking comparison, show the latency report, show a guardrail rejecting a bad query
- [ ] Final smoke test on a clean environment (fresh clone/install run)

**Deliverable:** Submission-ready repo + demo video + `README.md`.

**Exit criteria:** A stranger could clone the repo, follow the README, and reproduce the demo query.

---

## Cross-cutting notes (apply throughout)

- Keep every stage's fallback behavior explicit — never let a stage crash the whole pipeline silently.
- Log everything from Day 1 (even rough logs) — Phase 7's analytics depend on having real data to look back on, not just the final day's runs.
- Treat every "baseline vs advanced" comparison (STT, chunking) as submission content, not throwaway scratch work — write results down as you go.
