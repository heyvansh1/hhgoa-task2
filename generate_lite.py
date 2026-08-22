"""Lightweight LLM generation via Groq API.

Uses the Groq API to generate grounded answers from retrieved chunks.
Falls back to extractive answer if API is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
_FALLBACK = "I do not have sufficient information to answer this question."

_SYSTEM_PROMPT = """You are a helpful multilingual QA assistant. Answer the user's question ONLY using the retrieved context below. Be concise (2-3 sentences max).

Rules:
1. ONLY use information from the provided context.
2. If the context does not contain the answer, say "I could not find relevant information in the knowledge base."
3. Do NOT hallucinate or add information not in the context.
4. If the question is in Hindi, answer in Hindi. If in English, answer in English.
5. Do not mention "context" or "retrieved passages" in your answer — speak naturally."""


def generate_answer(
    query: str,
    chunks: list,
    language: str = "eng",
) -> str:
    """Generate a grounded answer using Groq API.

    Args:
        query: User's question.
        chunks: Retrieved chunks (objects with .text and .chunk_id attributes).
        language: "eng" or "hin".

    Returns:
        Generated answer string.
    """
    if not chunks:
        return _FALLBACK

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, using extractive fallback")
        return _extractive_fallback(chunks)

    # Build context from chunks
    context_parts = []
    for i, c in enumerate(chunks[:5], 1):
        text = getattr(c, "text", str(c))
        cid = getattr(c, "chunk_id", f"chunk_{i}")
        context_parts.append(f"[{i}] {text}")
    context = "\n\n".join(context_parts)

    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        last_error = ""
        for model in _MODELS:
            resp = requests.post(
                _GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                last_error = f"Groq API ({model}) error {resp.status_code}: {resp.text[:100]}"
                logger.warning(last_error)
                # If model not found, try next model
                if resp.status_code == 404:
                    continue
                # For other errors (rate limit, auth), don't retry
                break

        logger.warning("All Groq models failed. Last error: %s", last_error)
        return _extractive_fallback(chunks)

    except Exception as exc:
        logger.warning("Groq API call failed: %s", exc)
        return _extractive_fallback(chunks)


def _extractive_fallback(chunks: list) -> str:
    """Simple extractive answer from top chunk (no API needed)."""
    top = chunks[0]
    text = getattr(top, "text", str(top))
    return text
