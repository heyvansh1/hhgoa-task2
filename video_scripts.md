# Video Scripts

## Video 1: Team & Process (90 seconds)

### Script

**[0:00 – 0:10] Opening**
> "Hi, we're [Team Name], and we built a voice-enabled multilingual RAG system that processes English and Hindi audio queries against MSMARCO-XI — with sub-200ms retrieval and built-in hallucination guardrails."

**[0:10 – 0:30] Architecture & Key Decisions**
> "Our pipeline flows: Audio → Sarvam STT → Query Preprocessing → Hybrid Retrieval → Guardrails → LLM Generation.
>
> We chose Sarvam's Saarika API for STT because of its native Hindi support. For retrieval, we use in-process hnswlib for dense search plus BM25 for sparse keyword matching, fused with Reciprocal Rank Fusion."

**[0:30 – 0:55] Chunking Trade-offs (Phase 4 Benchmark)**
> "We benchmarked four chunking strategies: Passage-Native, Fixed-Size, Semantic, and Hierarchical.
>
> [Show chunking_comparison.png]
>
> Our benchmark on 100 queries showed that [best strategy] achieved the highest Recall@5 at [X]%, while maintaining [Y]ms average latency. We selected this as our default strategy because [rationale]."

**[0:55 – 1:20] Latency & Guardrails**
> "Our latency report shows retrieval P50 under 200ms. The full text-to-text pipeline runs under [X]ms P50.
>
> [Show latency_breakdown.png]
>
> For guardrails, we reject 80%+ of adversarial queries upfront, and verify answer groundedness via token overlap before returning any response."

**[1:20 – 1:30] Closing**
> "All code is modular, typed, and documented. The repo includes a one-command demo, a full benchmark suite, and honest latency numbers from real runs. Thank you."

---

## Video 2: Product Demo (2 minutes)

### Script

**[0:00 – 0:10] Opening**
> "Let me show you our voice RAG pipeline in action. We'll run three queries: a successful English query, a Hindi query, and an out-of-domain rejection."

**[0:10 – 0:40] Demo 1: Successful English Query**
> "First, let's ask in English."
>
> ```
> python main.py --query "What is the theory of relativity?" --lang eng
> ```
>
> [Show terminal output]
>
> "The pipeline preprocessed the query in [X]ms, retrieved 5 relevant chunks in [Y]ms, and generated a context-grounded answer. Total latency: [Z]ms.
>
> Notice the stage-by-stage timing breakdown — every stage is individually instrumented."

**[0:40 – 1:10] Demo 2: Successful Hindi Query**
> "Now let's try Hindi."
>
> ```
> python main.py --query "भारत की राजधानी क्या है?" --lang hin
> ```
>
> [Show terminal output]
>
> "Same pipeline, same latency target. The retriever searched the Hindi language partition specifically — this is our metadata pre-filtering in action. The answer correctly identifies New Delhi as India's capital."

**[1:10 – 1:35] Demo 3: Guardrail Rejection**
> "What happens with an off-topic query?"
>
> ```
> python main.py --query "Tell me a recipe for chocolate cake" --lang eng
> ```
>
> [Show terminal output]
>
> "The input guardrail caught this as off-topic before retrieval even ran. The pipeline returned a polite refusal in [X]ms — no wasted computation, no hallucinated cooking instructions."

**[1:35 – 1:50] Benchmark Results**
> "Let me show you our benchmark numbers."
>
> [Show chunking_benchmark_results.md table]
> [Show chunking_comparison.png chart]
>
> "Four chunking strategies, 100 queries, honest numbers. [Strategy] won on recall, [Strategy] won on latency."

**[1:50 – 2:00] Closing**
> "Everything you've seen runs locally, in-process, with no external vector DB. Clone the repo, install requirements, and run `python main.py` — it works out of the box. Thank you."

---

## Recording Notes

- Record terminal at 1080p with a clean, dark-themed terminal
- Use a legible monospace font (e.g., JetBrains Mono, 16pt)
- Pre-run the model download so the demo doesn't wait for downloads
- Keep pauses short between commands — the timer is tight
- For Video 2, consider a split-screen with terminal + code editor showing the pipeline
