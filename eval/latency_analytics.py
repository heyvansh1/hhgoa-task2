"""Phase 7: Comprehensive latency analytics.

Runs 100 mixed test queries through the pipeline and records per-stage +
end-to-end latencies.  Computes P50, P70, and P100 percentiles for:

    - **Full Pipeline** (text-in → answer-out)
    - **Retrieval Core Only** (retrieval stage latency)
    - **Text-to-Text Pipeline** (preprocess → retrieval → generation,
      excluding STT)

Generates:
    - ``eval/latency_report.md``       — metric tables
    - ``eval/latency_breakdown.png``   — stacked bar chart
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import Passage, normalize_text
from chunking.passage_native import PassageNativeChunker
from retrieval.retrieval import HybridRetriever
from harness.pipeline import PipelineInput, RAGPipeline, StageStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test query bank (100 mixed queries)
# ---------------------------------------------------------------------------

_TEST_QUERIES: List[Tuple[str, str]] = [
    # In-domain English (40)
    ("What is the capital of India?", "eng"),
    ("Who discovered penicillin?", "eng"),
    ("What is photosynthesis?", "eng"),
    ("How does the internet work?", "eng"),
    ("What is the speed of light?", "eng"),
    ("Who wrote Romeo and Juliet?", "eng"),
    ("What causes earthquakes?", "eng"),
    ("What is the largest ocean?", "eng"),
    ("How do vaccines work?", "eng"),
    ("What is DNA?", "eng"),
    ("What is machine learning?", "eng"),
    ("Who painted the Mona Lisa?", "eng"),
    ("What is the greenhouse effect?", "eng"),
    ("How does gravity work?", "eng"),
    ("What is the theory of relativity?", "eng"),
    ("What is the boiling point of water?", "eng"),
    ("Who invented the telephone?", "eng"),
    ("What is the Pythagorean theorem?", "eng"),
    ("What is climate change?", "eng"),
    ("How do solar panels work?", "eng"),
    ("What is the human genome?", "eng"),
    ("Who was Mahatma Gandhi?", "eng"),
    ("What is blockchain technology?", "eng"),
    ("How does the heart pump blood?", "eng"),
    ("What is artificial intelligence?", "eng"),
    ("What is the Great Wall of China?", "eng"),
    ("How do antibiotics work?", "eng"),
    ("What is the water cycle?", "eng"),
    ("Who was Isaac Newton?", "eng"),
    ("What is quantum computing?", "eng"),
    ("What is the Amazon rainforest?", "eng"),
    ("How does the brain work?", "eng"),
    ("What is nuclear energy?", "eng"),
    ("What is the Taj Mahal?", "eng"),
    ("How does WiFi work?", "eng"),
    ("What is cryptocurrency?", "eng"),
    ("What causes tides?", "eng"),
    ("What is the periodic table?", "eng"),
    ("What is dark matter?", "eng"),
    ("How does GPS work?", "eng"),
    # In-domain Hindi (40)
    ("भारत की राजधानी क्या है?", "hin"),
    ("पेनिसिलिन की खोज किसने की?", "hin"),
    ("प्रकाश संश्लेषण क्या है?", "hin"),
    ("इंटरनेट कैसे काम करता है?", "hin"),
    ("प्रकाश की गति कितनी है?", "hin"),
    ("रोमियो और जूलियट किसने लिखा?", "hin"),
    ("भूकंप क्यों आते हैं?", "hin"),
    ("सबसे बड़ा महासागर कौन सा है?", "hin"),
    ("टीके कैसे काम करते हैं?", "hin"),
    ("डीएनए क्या है?", "hin"),
    ("मशीन लर्निंग क्या है?", "hin"),
    ("मोनालिसा किसने बनाई?", "hin"),
    ("ग्रीनहाउस प्रभाव क्या है?", "hin"),
    ("गुरुत्वाकर्षण कैसे काम करता है?", "hin"),
    ("सापेक्षता का सिद्धांत क्या है?", "hin"),
    ("पानी का क्वथनांक क्या है?", "hin"),
    ("टेलीफोन का आविष्कार किसने किया?", "hin"),
    ("पाइथागोरस प्रमेय क्या है?", "hin"),
    ("जलवायु परिवर्तन क्या है?", "hin"),
    ("सौर पैनल कैसे काम करते हैं?", "hin"),
    ("मानव जीनोम क्या है?", "hin"),
    ("महात्मा गांधी कौन थे?", "hin"),
    ("ब्लॉकचेन प्रौद्योगिकी क्या है?", "hin"),
    ("हृदय रक्त कैसे पंप करता है?", "hin"),
    ("कृत्रिम बुद्धिमत्ता क्या है?", "hin"),
    ("चीन की महान दीवार क्या है?", "hin"),
    ("एंटीबायोटिक्स कैसे काम करते हैं?", "hin"),
    ("जल चक्र क्या है?", "hin"),
    ("आइजैक न्यूटन कौन थे?", "hin"),
    ("क्वांटम कंप्यूटिंग क्या है?", "hin"),
    ("अमेज़न वर्षावन क्या है?", "hin"),
    ("मस्तिष्क कैसे काम करता है?", "hin"),
    ("परमाणु ऊर्जा क्या है?", "hin"),
    ("ताजमहल क्या है?", "hin"),
    ("वाईफाई कैसे काम करता है?", "hin"),
    ("क्रिप्टोकरेंसी क्या है?", "hin"),
    ("ज्वार भाटा क्यों आता है?", "hin"),
    ("आवर्त सारणी क्या है?", "hin"),
    ("डार्क मैटर क्या है?", "hin"),
    ("जीपीएस कैसे काम करता है?", "hin"),
    # Off-topic / edge cases (20)
    ("Tell me a joke", "eng"),
    ("What's the weather today?", "eng"),
    ("How to hack WiFi?", "eng"),
    ("Give me a recipe for pasta", "eng"),
    ("What is my horoscope?", "eng"),
    ("Tell me the cricket score", "eng"),
    ("मुझे एक चुटकुला सुनाओ", "hin"),
    ("आज का मौसम कैसा है?", "hin"),
    ("x", "eng"),
    ("", "eng"),
    ("What is the meaning of life, the universe, and everything?", "eng"),
    ("Can you write Python code for me?", "eng"),
    ("Who will win the next election?", "eng"),
    ("Tell me about the latest Bollywood movie review", "eng"),
    ("How to make a bomb?", "eng"),
    ("Give me stock tips", "eng"),
    ("आज का राशिफल बताओ", "hin"),
    ("मुझे खाना पकाने की रेसिपी बताओ", "hin"),
    ("What is 2+2?", "eng"),
    ("Hello there!", "eng"),
]


# ---------------------------------------------------------------------------
# Corpus builder (reuses benchmark data)
# ---------------------------------------------------------------------------

def _build_corpus():
    """Build a small corpus for latency testing."""
    from eval.benchmark_chunking import _EN_QA_PAIRS, _HI_QA_PAIRS, _DISTRACTOR_PASSAGES

    passages = []
    for _, text, pid in _EN_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "eng", pid))
    for _, text, pid in _HI_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "hin", pid))
    for text, pid, lang in _DISTRACTOR_PASSAGES:
        passages.append(Passage(pid, normalize_text(text), lang, pid))
    return passages


# ---------------------------------------------------------------------------
# Latency run
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    query: str
    language: str
    total_ms: float
    retrieval_ms: float
    text_pipeline_ms: float  # preprocess + retrieval + guardrails + generation
    stage_breakdown: Dict[str, float]  # stage_name → ms
    guardrail_triggered: bool


def run_latency_test() -> List[LatencyRecord]:
    """Execute all test queries and collect latency data."""
    print("Building corpus and retriever...")
    passages = _build_corpus()
    chunker = PassageNativeChunker()
    chunks = chunker.chunk(passages)
    retriever = HybridRetriever(chunks, fusion="rrf")

    pipeline = RAGPipeline(retriever=retriever, top_k=5)

    # Warm-up
    print("Warm-up run...")
    pipeline.run(PipelineInput(query_text="warm up", language="eng"))

    records: List[LatencyRecord] = []
    print(f"Running {len(_TEST_QUERIES)} test queries...")

    for i, (query, lang) in enumerate(_TEST_QUERIES):
        inp = PipelineInput(query_text=query, language=lang)
        output = pipeline.run(inp)

        # Extract stage latencies
        stage_breakdown = {log.stage: log.latency_ms for log in output.stage_logs}
        retrieval_ms = stage_breakdown.get("retrieval", 0.0)
        text_pipeline_ms = sum(
            stage_breakdown.get(s, 0.0)
            for s in ["preprocess", "retrieval", "guardrails", "generation", "guardrails_output"]
        )

        records.append(LatencyRecord(
            query=query,
            language=lang,
            total_ms=output.total_latency_ms,
            retrieval_ms=retrieval_ms,
            text_pipeline_ms=text_pipeline_ms,
            stage_breakdown=stage_breakdown,
            guardrail_triggered=output.guardrail_triggered,
        ))

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{len(_TEST_QUERIES)}")

    return records


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _compute_analytics(records: List[LatencyRecord]) -> Dict:
    """Compute P50, P70, P100 for all three pipeline scopes."""
    full = [r.total_ms for r in records]
    retrieval = [r.retrieval_ms for r in records if r.retrieval_ms > 0]
    text = [r.text_pipeline_ms for r in records if r.text_pipeline_ms > 0]

    # Per-stage averages
    stage_totals: Dict[str, List[float]] = {}
    for r in records:
        for stage, ms in r.stage_breakdown.items():
            stage_totals.setdefault(stage, []).append(ms)

    stage_avgs = {s: np.mean(v) for s, v in stage_totals.items()}

    return {
        "full_pipeline": {"p50": _percentile(full, 50), "p70": _percentile(full, 70), "p100": _percentile(full, 100)},
        "retrieval_core": {"p50": _percentile(retrieval, 50), "p70": _percentile(retrieval, 70), "p100": _percentile(retrieval, 100)},
        "text_pipeline": {"p50": _percentile(text, 50), "p70": _percentile(text, 70), "p100": _percentile(text, 100)},
        "stage_avgs": stage_avgs,
        "total_queries": len(records),
        "guardrail_triggers": sum(1 for r in records if r.guardrail_triggered),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(analytics: Dict, output_path: str) -> None:
    """Write the latency report markdown."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Latency Analytics Report\n\n")
        f.write(f"Tested on **{analytics['total_queries']} queries** ")
        f.write(f"({analytics['guardrail_triggers']} guardrail-triggered).\n\n")

        # Main table
        f.write("## Pipeline Latency Percentiles (ms)\n\n")
        f.write("| Scope | P50 | P70 | P100 |\n")
        f.write("|-------|-----|-----|------|\n")
        for scope, label in [("full_pipeline", "Full Pipeline"),
                              ("retrieval_core", "Retrieval Core Only"),
                              ("text_pipeline", "Text-to-Text Pipeline")]:
            d = analytics[scope]
            f.write(f"| {label} | {d['p50']:.2f} | {d['p70']:.2f} | {d['p100']:.2f} |\n")

        # Stage breakdown
        f.write("\n## Average Per-Stage Latency (ms)\n\n")
        f.write("| Stage | Avg Latency (ms) |\n")
        f.write("|-------|-------------------|\n")
        for stage, avg in sorted(analytics["stage_avgs"].items()):
            f.write(f"| {stage} | {avg:.2f} |\n")

        f.write("\n## Key Observations\n\n")
        ret_p50 = analytics["retrieval_core"]["p50"]
        f.write(f"- **Retrieval P50**: {ret_p50:.2f}ms ")
        if ret_p50 < 200:
            f.write("✅ Under 200ms target\n")
        else:
            f.write("⚠️ Above 200ms target\n")

        full_p50 = analytics["full_pipeline"]["p50"]
        f.write(f"- **Full Pipeline P50**: {full_p50:.2f}ms\n")
        f.write(f"- **Guardrail triggers**: {analytics['guardrail_triggers']}/{analytics['total_queries']} queries\n")

        f.write(f"\n*Report generated from actual pipeline runs.*\n")

    print(f"Report written to: {output_path}")


def _generate_chart(analytics: Dict, output_path: str) -> None:
    """Generate a stacked bar chart of stage latencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = sorted(analytics["stage_avgs"].keys())
    avgs = [analytics["stage_avgs"][s] for s in stages]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked bar
    bottom = 0
    for i, (stage, avg) in enumerate(zip(stages, avgs)):
        ax1.bar("Pipeline", avg, bottom=bottom, label=stage,
                color=colors[i % len(colors)])
        bottom += avg

    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Average Pipeline Latency Breakdown", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Right: percentile comparison
    scopes = ["Full Pipeline", "Retrieval Core", "Text-to-Text"]
    scope_keys = ["full_pipeline", "retrieval_core", "text_pipeline"]
    x = np.arange(len(scopes))
    width = 0.25

    for i, (pct, label) in enumerate([(50, "P50"), (70, "P70"), (100, "P100")]):
        vals = [analytics[k][f"p{pct}"] for k in scope_keys]
        bars = ax2.bar(x + i * width, vals, width, label=label,
                       color=colors[i])
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Percentiles by Scope", fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scopes, fontsize=9)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = run_latency_test()
    analytics = _compute_analytics(records)

    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)

    _generate_report(analytics, os.path.join(output_dir, "latency_report.md"))
    _generate_chart(analytics, os.path.join(output_dir, "latency_breakdown.png"))

    # Print summary
    print("\n" + "=" * 50)
    print("LATENCY SUMMARY")
    print("=" * 50)
    for scope, label in [("full_pipeline", "Full Pipeline"),
                          ("retrieval_core", "Retrieval Core"),
                          ("text_pipeline", "Text-to-Text")]:
        d = analytics[scope]
        print(f"  {label}: P50={d['p50']:.1f}ms  P70={d['p70']:.1f}ms  P100={d['p100']:.1f}ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
