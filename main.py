"""Voice RAG Pipeline — CLI Demo Script.

Usage::

    # Text query (English)
    python main.py --query "What is the capital of India?" --lang eng

    # Text query (Hindi)
    python main.py --query "भारत की राजधानी क्या है?" --lang hin

    # Audio query
    python main.py --audio path/to/audio.wav --lang hin

    # Run benchmark
    python main.py --benchmark

    # Run latency analytics
    python main.py --latency
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from data.preprocess import Passage, normalize_text


def _build_retriever():
    """Build corpus and hybrid retriever from benchmark data."""
    from eval.benchmark_chunking import _EN_QA_PAIRS, _HI_QA_PAIRS, _DISTRACTOR_PASSAGES
    from chunking.passage_native import PassageNativeChunker
    from retrieval.retrieval import HybridRetriever

    passages = []
    for _, text, pid in _EN_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "eng", pid))
    for _, text, pid in _HI_QA_PAIRS:
        passages.append(Passage(pid, normalize_text(text), "hin", pid))
    for text, pid, lang in _DISTRACTOR_PASSAGES:
        passages.append(Passage(pid, normalize_text(text), lang, pid))

    chunker = PassageNativeChunker()
    chunks = chunker.chunk(passages)
    return HybridRetriever(chunks, fusion="rrf")


def _print_header(text: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def _print_stage_log(log) -> None:
    status_icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(log.status, "?")
    print(f"  {status_icon} {log.stage:<20s}  {log.latency_ms:>8.2f}ms  {log.status}")
    if log.error:
        print(f"     ⚠ {log.error}")


def run_query(query: str, lang: str, audio_path: str = None) -> None:
    """Run a single query through the pipeline and pretty-print results."""
    from harness.pipeline import PipelineInput, RAGPipeline

    _print_header("VOICE RAG PIPELINE")
    print(f"  Query: {query or '(audio input)'}")
    print(f"  Language: {lang}")
    if audio_path:
        print(f"  Audio: {audio_path}")

    print("\n  ⏳ Building retriever...")
    start = time.perf_counter()
    retriever = _build_retriever()
    build_time = (time.perf_counter() - start) * 1000
    print(f"  ✅ Retriever built in {build_time:.0f}ms")

    pipeline = RAGPipeline(retriever=retriever, top_k=5)
    inp = PipelineInput(query_text=query, audio_path=audio_path, language=lang)

    _print_header("EXECUTING PIPELINE")
    output = pipeline.run(inp)

    # Stage logs
    _print_header("STAGE TIMINGS")
    for log in output.stage_logs:
        _print_stage_log(log)
    print(f"\n  {'─' * 40}")
    print(f"  Total latency: {output.total_latency_ms:.2f}ms")

    # Retrieved chunks
    if output.chunks_cited:
        _print_header("RETRIEVED CHUNKS")
        for i, chunk in enumerate(output.chunks_cited[:5], 1):
            text_preview = chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text
            print(f"  [{i}] {chunk.chunk_id} ({chunk.language})")
            print(f"      {text_preview}")

    # Answer
    _print_header("ANSWER")
    if output.guardrail_triggered:
        print(f"  🛡️ GUARDRAIL TRIGGERED")
    print(f"  {output.answer}")
    print()


def run_benchmark() -> None:
    """Run the chunking benchmark."""
    _print_header("RUNNING CHUNKING BENCHMARK")
    from eval.benchmark_chunking import main as bench_main
    bench_main()
    print("\n✅ Benchmark complete. Check eval/chunking_benchmark_results.md")


def run_latency() -> None:
    """Run latency analytics."""
    _print_header("RUNNING LATENCY ANALYTICS")
    from eval.latency_analytics import main as lat_main
    lat_main()
    print("\n✅ Latency analysis complete. Check eval/latency_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice RAG Pipeline — Multilingual QA on MSMARCO-XI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What is the capital of India?" --lang eng
  python main.py --query "भारत की राजधानी क्या है?" --lang hin
  python main.py --benchmark
  python main.py --latency
        """,
    )
    parser.add_argument("--query", "-q", type=str, help="Text query to process")
    parser.add_argument("--audio", "-a", type=str, help="Path to audio file (.wav)")
    parser.add_argument("--lang", "-l", type=str, default="eng",
                        choices=["eng", "hin"], help="Query language (default: eng)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the chunking strategy benchmark")
    parser.add_argument("--latency", action="store_true",
                        help="Run latency analytics")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.benchmark:
        run_benchmark()
    elif args.latency:
        run_latency()
    elif args.query or args.audio:
        run_query(args.query or "", args.lang, args.audio)
    else:
        parser.print_help()
        print("\n💡 Try: python main.py --query \"What is the capital of India?\" --lang eng")


if __name__ == "__main__":
    main()
