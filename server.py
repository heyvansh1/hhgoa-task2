"""Lightweight FastAPI server for Render free-tier deployment.

Endpoints:
    GET  /           → Serves the chat UI
    POST /api/query  → Text query  → retrieval + generation
    POST /api/voice  → Audio upload → STT + retrieval + generation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from retrieval_lite import LiteRetriever
from generate_lite import generate_answer
from guardrails.guardrails import validate_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Resolve paths (absolute — works regardless of CWD) ─────────────

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# ─── Initialise retriever (instant — no ML model loading) ────────────

print("[*] Building TF-IDF index...")
_start = time.perf_counter()
retriever = LiteRetriever()
print(f"[OK] Index built in {(time.perf_counter() - _start)*1000:.0f}ms")

# ─── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(title="Voice RAG Pipeline", version="1.0.0")


# ─── Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    language: str = "eng"


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    language: str


class QueryResponse(BaseModel):
    answer: str
    query_text: str
    language: str
    retrieval_latency_ms: float
    total_latency_ms: float
    guardrail_triggered: bool = False
    chunks: List[ChunkResponse] = []


# ─── Routes ──────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(str(INDEX_HTML))


@app.head("/")
async def health_check():
    """Render health check sends HEAD /. Return 200 OK."""
    return Response(status_code=200)


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    pipeline_start = time.perf_counter()

    query = req.query.strip()
    language = req.language

    # Preprocess
    query = unicodedata.normalize("NFKC", query)
    query = re.sub(r"\s+", " ", query).strip()

    if not query:
        return QueryResponse(
            answer="Please enter a question.",
            query_text=query,
            language=language,
            retrieval_latency_ms=0,
            total_latency_ms=0,
        )

    # Input guardrails
    passed, reason = validate_input(query)
    if not passed:
        return QueryResponse(
            answer=reason,
            query_text=query,
            language=language,
            retrieval_latency_ms=0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
            guardrail_triggered=True,
        )

    # Retrieval
    ret_start = time.perf_counter()
    chunks = retriever.retrieve(query, k=5, lang=language)
    retrieval_ms = (time.perf_counter() - ret_start) * 1000

    # Generation
    answer = generate_answer(query, chunks, language)

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    return QueryResponse(
        answer=answer,
        query_text=query,
        language=language,
        retrieval_latency_ms=round(retrieval_ms, 2),
        total_latency_ms=round(total_ms, 2),
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                text=c.text[:200],
                language=c.language,
            )
            for c in chunks[:3]
        ],
    )


@app.post("/api/voice", response_model=QueryResponse)
async def handle_voice(
    audio: UploadFile = File(...),
    language: str = Form("eng"),
):
    pipeline_start = time.perf_counter()

    # Save uploaded audio to temp file
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        audio_path = tmp.name

    # STT via Sarvam API
    try:
        from stt.sarvam import SarvamSTT
        stt = SarvamSTT()
        result = stt.transcribe(audio_path)
        if result.error:
            return QueryResponse(
                answer=f"STT Error: {result.error}",
                query_text="",
                language=language,
                retrieval_latency_ms=0,
                total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
            )
        query = result.transcript
    except Exception as e:
        return QueryResponse(
            answer=f"Could not process audio: {str(e)}",
            query_text="",
            language=language,
            retrieval_latency_ms=0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
        )
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    # Preprocess
    query = unicodedata.normalize("NFKC", query)
    query = re.sub(r"\s+", " ", query).strip()

    if not query:
        return QueryResponse(
            answer="Could not understand the audio. Please try again.",
            query_text="",
            language=language,
            retrieval_latency_ms=0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
        )

    # Input guardrails
    passed, reason = validate_input(query)
    if not passed:
        return QueryResponse(
            answer=reason,
            query_text=query,
            language=language,
            retrieval_latency_ms=0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
            guardrail_triggered=True,
        )

    # Retrieval
    ret_start = time.perf_counter()
    chunks = retriever.retrieve(query, k=5, lang=language)
    retrieval_ms = (time.perf_counter() - ret_start) * 1000

    # Generation
    answer = generate_answer(query, chunks, language)

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    return QueryResponse(
        answer=answer,
        query_text=query,
        language=language,
        retrieval_latency_ms=round(retrieval_ms, 2),
        total_latency_ms=round(total_ms, 2),
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                text=c.text[:200],
                language=c.language,
            )
            for c in chunks[:3]
        ],
    )


# ─── Static files (mount AFTER routes so API routes take priority) ───

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
