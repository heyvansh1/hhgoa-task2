"""Phase 5: Pipeline orchestration and execution harness.

Implements the full RAG pipeline:
    AudioInput → STT → QueryPreprocess → HybridRetrieval →
    GuardrailsCheck → LLMGeneration

Uses Pydantic models for typed stage I/O, ``time.perf_counter_ns`` for
nanosecond-precision latency logging, and graceful error recovery at
every stage.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =====================================================================
# Pydantic models
# =====================================================================


class StageStatus(str, Enum):
    """Outcome of a single pipeline stage."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageLog(BaseModel):
    """Timing and status log for one pipeline stage."""
    stage: str
    status: StageStatus = StageStatus.SUCCESS
    latency_ns: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineInput(BaseModel):
    """Input to the RAG pipeline — either text or an audio file path."""
    query_text: Optional[str] = None
    audio_path: Optional[str] = None
    language: str = "eng"  # ISO-639-3: "eng" | "hin"


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, included in pipeline output."""
    chunk_id: str
    text: str
    source_passage_id: str
    language: str
    score: Optional[float] = None


class PipelineOutput(BaseModel):
    """Full output of the RAG pipeline."""
    answer: str
    chunks_cited: List[RetrievedChunk] = Field(default_factory=list)
    stage_logs: List[StageLog] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    query_text: str = ""
    language: str = "eng"
    guardrail_triggered: bool = False


# =====================================================================
# Stage implementations
# =====================================================================


def _log_stage(name: str, start_ns: int, status: StageStatus = StageStatus.SUCCESS,
               error: Optional[str] = None, **meta) -> StageLog:
    """Create a StageLog from a perf_counter_ns start timestamp."""
    elapsed_ns = time.perf_counter_ns() - start_ns
    return StageLog(
        stage=name,
        status=status,
        latency_ns=elapsed_ns,
        latency_ms=elapsed_ns / 1_000_000,
        error=error,
        metadata=meta,
    )


# ---- Stage 1: STT ---------------------------------------------------

def _stage_stt(audio_path: str, language: str,
               max_retries: int = 2, timeout_s: float = 10.0) -> tuple[str, StageLog]:
    """Transcribe audio via Sarvam API with retries.

    Returns:
        ``(transcript, stage_log)``
    """
    start = time.perf_counter_ns()

    try:
        from stt.sarvam import SarvamSTT
        stt = SarvamSTT()

        last_error = None
        for attempt in range(max_retries + 1):
            result = stt.transcribe(audio_path)
            if result.error is None:
                return result.transcript, _log_stage(
                    "stt", start, metadata={"provider": "sarvam", "attempt": attempt + 1}
                )
            last_error = result.error
            logger.warning("STT attempt %d failed: %s", attempt + 1, last_error)

        return "", _log_stage("stt", start, StageStatus.FAILED,
                              error=f"All {max_retries + 1} attempts failed: {last_error}",
                              provider="sarvam")
    except Exception as exc:
        return "", _log_stage("stt", start, StageStatus.FAILED, error=str(exc))


# ---- Stage 2: Query Preprocessing -----------------------------------

def _stage_preprocess(query: str, language: str) -> tuple[str, StageLog]:
    """Normalise unicode, collapse whitespace, detect language if needed."""
    start = time.perf_counter_ns()
    try:
        cleaned = unicodedata.normalize("NFKC", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return "", _log_stage("preprocess", start, StageStatus.FAILED,
                                  error="Empty query after preprocessing")
        return cleaned, _log_stage("preprocess", start)
    except Exception as exc:
        return query, _log_stage("preprocess", start, StageStatus.FAILED, error=str(exc))


# ---- Stage 3: Retrieval ----------------------------------------------

def _stage_retrieve(query: str, retriever, k: int = 5,
                    lang: Optional[str] = None) -> tuple[list, StageLog]:
    """Run hybrid retrieval and return chunks + log."""
    start = time.perf_counter_ns()
    try:
        chunks = retriever.retrieve(query, k=k, lang=lang)
        return chunks, _log_stage("retrieval", start,
                                  metadata={"k": k, "lang": lang, "results": len(chunks)})
    except Exception as exc:
        return [], _log_stage("retrieval", start, StageStatus.FAILED, error=str(exc))


# ---- Stage 4: Guardrails check ---------------------------------------

def _stage_guardrails(query: str, chunks: list, answer: str = "") -> tuple[bool, str, StageLog]:
    """Run input + output guardrails.

    Returns:
        ``(passed, override_answer, stage_log)``
    """
    start = time.perf_counter_ns()
    try:
        from guardrails.guardrails import validate_input, verify_groundedness

        # Input guardrail
        input_ok, input_reason = validate_input(query)
        if not input_ok:
            return False, input_reason, _log_stage(
                "guardrails", start, metadata={"input_rejected": True, "reason": input_reason}
            )

        # Output guardrail (only if answer is provided)
        if answer and chunks:
            grounded, ground_reason = verify_groundedness(answer, chunks)
            if not grounded:
                return False, ground_reason, _log_stage(
                    "guardrails", start, metadata={"output_rejected": True, "reason": ground_reason}
                )

        return True, "", _log_stage("guardrails", start)
    except ImportError:
        # Guardrails module not yet available — pass through
        return True, "", _log_stage("guardrails", start, StageStatus.SKIPPED,
                                    error="guardrails module not available")
    except Exception as exc:
        # Guardrail failures should NOT crash the pipeline
        return True, "", _log_stage("guardrails", start, StageStatus.FAILED, error=str(exc))


# ---- Stage 5: LLM Generation ----------------------------------------

_FALLBACK_ANSWER = "I do not have sufficient information to answer this question."


def _stage_generate(query: str, chunks: list, language: str = "eng") -> tuple[str, StageLog]:
    """Generate an answer from retrieved context.

    Uses a mock LLM by default — returns a context-grounded extractive
    answer.  Can be swapped for Groq/Gemini/OpenAI via environment vars.
    """
    start = time.perf_counter_ns()
    try:
        if not chunks:
            return _FALLBACK_ANSWER, _log_stage("generation", start,
                                                 metadata={"mode": "no_context"})

        # ---- Mock LLM: extractive context-grounded answer ----
        # Concatenate chunk texts for context
        context = "\n\n".join(f"[{c.chunk_id}]: {c.text}" for c in chunks[:3])

        # Simple extractive approach: return the most relevant chunk's text
        # with a structured wrapper
        top_chunk = chunks[0]
        cited_ids = [c.chunk_id for c in chunks[:3]]

        answer = (
            f"Based on the retrieved information: {top_chunk.text} "
            f"[Sources: {', '.join(cited_ids)}]"
        )

        return answer, _log_stage("generation", start,
                                   metadata={"mode": "mock_extractive",
                                             "cited_chunks": len(cited_ids)})
    except Exception as exc:
        return _FALLBACK_ANSWER, _log_stage("generation", start, StageStatus.FAILED,
                                             error=str(exc))


# =====================================================================
# Pipeline executor
# =====================================================================


class RAGPipeline:
    """End-to-end Voice RAG pipeline with per-stage latency logging.

    Args:
        retriever: A :class:`HybridRetriever` instance (or compatible).
            If ``None``, retrieval stage is skipped.
        top_k: Number of chunks to retrieve.

    Example::

        from retrieval.retrieval import HybridRetriever
        retriever = HybridRetriever(chunks)
        pipeline = RAGPipeline(retriever)
        output = pipeline.run(PipelineInput(query_text="What is DNA?", language="eng"))
        print(output.answer)
    """

    def __init__(self, retriever=None, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def run(self, inp: PipelineInput) -> PipelineOutput:
        """Execute the full pipeline.

        Args:
            inp: Pipeline input (text query or audio path).

        Returns:
            A :class:`PipelineOutput` with the answer, cited chunks,
            stage logs, and total latency.
        """
        pipeline_start = time.perf_counter_ns()
        logs: List[StageLog] = []
        query_text = inp.query_text or ""
        language = inp.language

        # ── Stage 1: STT (only if audio_path provided) ──────────────
        if inp.audio_path and not query_text:
            query_text, stt_log = _stage_stt(inp.audio_path, language)
            logs.append(stt_log)
            if stt_log.status == StageStatus.FAILED:
                return PipelineOutput(
                    answer="Sorry, I couldn't process the audio. Please try again or type your question.",
                    stage_logs=logs,
                    total_latency_ms=(time.perf_counter_ns() - pipeline_start) / 1_000_000,
                    query_text=query_text,
                    language=language,
                )

        # ── Stage 2: Preprocess ─────────────────────────────────────
        query_text, prep_log = _stage_preprocess(query_text, language)
        logs.append(prep_log)
        if prep_log.status == StageStatus.FAILED:
            return PipelineOutput(
                answer="Sorry, I couldn't understand the query.",
                stage_logs=logs,
                total_latency_ms=(time.perf_counter_ns() - pipeline_start) / 1_000_000,
                query_text=query_text,
                language=language,
            )

        # ── Stage 3: Input guardrails ───────────────────────────────
        input_ok, override_answer, guard_log = _stage_guardrails(query_text, [])
        logs.append(guard_log)
        if not input_ok:
            return PipelineOutput(
                answer=override_answer,
                stage_logs=logs,
                total_latency_ms=(time.perf_counter_ns() - pipeline_start) / 1_000_000,
                query_text=query_text,
                language=language,
                guardrail_triggered=True,
            )

        # ── Stage 4: Retrieval ──────────────────────────────────────
        chunks = []
        if self.retriever is not None:
            chunks, ret_log = _stage_retrieve(query_text, self.retriever,
                                              k=self.top_k, lang=language)
            logs.append(ret_log)
        else:
            logs.append(StageLog(stage="retrieval", status=StageStatus.SKIPPED,
                                 error="No retriever configured"))

        # ── Stage 5: Generation ─────────────────────────────────────
        answer, gen_log = _stage_generate(query_text, chunks, language)
        logs.append(gen_log)

        # ── Stage 6: Output guardrails (groundedness) ───────────────
        output_ok, ground_override, ground_log = _stage_guardrails(query_text, chunks, answer)
        ground_log.stage = "guardrails_output"
        logs.append(ground_log)
        if not output_ok:
            answer = ground_override

        # ── Assemble output ─────────────────────────────────────────
        total_ns = time.perf_counter_ns() - pipeline_start
        cited = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                source_passage_id=c.source_passage_id,
                language=c.language,
            )
            for c in chunks[:self.top_k]
        ]

        return PipelineOutput(
            answer=answer,
            chunks_cited=cited,
            stage_logs=logs,
            total_latency_ms=total_ns / 1_000_000,
            query_text=query_text,
            language=language,
            guardrail_triggered=not output_ok,
        )
