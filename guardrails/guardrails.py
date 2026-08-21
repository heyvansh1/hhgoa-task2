"""Phase 6: Guardrails — input validation and output groundedness checks.

Provides two main entry points:

    - :func:`validate_input` — fast keyword/regex + embedding-similarity
      check to reject off-topic or unsafe queries before retrieval runs.
    - :func:`verify_groundedness` — token/entity overlap check to ensure
      the generated answer is grounded in retrieved context.

Both return ``(passed: bool, reason: str)`` tuples.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# Constants
# =====================================================================

_REFUSAL = "I do not have sufficient information to answer this question."

# Keywords / regex patterns for unsafe or clearly off-topic input
_UNSAFE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(hack|exploit|crack|bypass|inject|malware|ransomware)\b", re.IGNORECASE),
    re.compile(r"\b(kill|murder|bomb|weapon|attack|terrorist)\b", re.IGNORECASE),
    re.compile(r"\b(porn|nude|sex|xxx|nsfw)\b", re.IGNORECASE),
    re.compile(r"\b(suicide|self[- ]?harm)\b", re.IGNORECASE),
    re.compile(r"\b(illegal drug|cocaine|heroin|meth|fentanyl)\b", re.IGNORECASE),
]

# Off-topic domain markers — queries about these are unlikely to be
# answerable by a factual knowledge corpus
_OFFTOPIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(recipe|cook|bake|ingredient|dessert)\b", re.IGNORECASE),
    re.compile(r"\b(movie review|film rating|box office|oscars|bollywood gossip)\b", re.IGNORECASE),
    re.compile(r"\b(horoscope|zodiac|astrology|kundli)\b", re.IGNORECASE),
    re.compile(r"\b(stock tip|invest|trading signal|buy sell)\b", re.IGNORECASE),
    re.compile(r"\b(joke|riddle|tell me something funny)\b", re.IGNORECASE),
    re.compile(r"\b(sports score|live match|cricket score|football result)\b", re.IGNORECASE),
    re.compile(r"\b(weather forecast|temperature today)\b", re.IGNORECASE),
    # Hindi off-topic
    re.compile(r"\b(रेसिपी|खाना बनाओ|पकाने)\b"),
    re.compile(r"\b(राशिफल|कुंडली|ज्योतिष)\b"),
    re.compile(r"\b(चुटकुला|मज़ाक|जोक)\b"),
    re.compile(r"\b(मौसम|तापमान आज)\b"),
]

# Minimum query length (too-short queries are likely noise)
_MIN_QUERY_LENGTH = 3

# Groundedness thresholds
_TOKEN_OVERLAP_THRESHOLD = 0.15   # At least 15% of answer tokens appear in context
_ENTITY_OVERLAP_THRESHOLD = 0.30  # At least 30% of answer "entities" appear in context


# =====================================================================
# Input validation
# =====================================================================


def validate_input(query: str) -> Tuple[bool, str]:
    """Fast input guardrail — reject unsafe or off-topic queries.

    Checks performed (in order):
        1. Minimum length check.
        2. Unsafe content pattern matching (keyword/regex blocklist).
        3. Off-topic domain pattern matching.

    Args:
        query: The user's query string (post-preprocessing).

    Returns:
        ``(passed, reason)`` — if ``passed`` is ``False``, ``reason``
        contains a human-readable rejection message.
    """
    if not query or len(query.strip()) < _MIN_QUERY_LENGTH:
        return False, _REFUSAL

    # 1. Unsafe content
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query):
            logger.info("Input guardrail: unsafe content detected — %s", pattern.pattern)
            return False, "I cannot assist with that type of request."

    # 2. Off-topic domain
    for pattern in _OFFTOPIC_PATTERNS:
        if pattern.search(query):
            logger.info("Input guardrail: off-topic query detected — %s", pattern.pattern)
            return False, _REFUSAL

    return True, ""


# =====================================================================
# Embedding-based topicality check (optional, heavier)
# =====================================================================


_CORPUS_CENTROID = None


def set_corpus_centroid(centroid) -> None:
    """Set the pre-computed corpus centroid for embedding-similarity checks.

    Call this once during pipeline initialisation with::

        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        centroid = np.mean(embedder.encode(all_chunk_texts), axis=0)
        set_corpus_centroid(centroid)
    """
    global _CORPUS_CENTROID
    _CORPUS_CENTROID = centroid


def validate_input_embedding(query: str, threshold: float = 0.25) -> Tuple[bool, str]:
    """Embedding-based input validation (slower, more accurate).

    Compares the query embedding's cosine similarity to the corpus centroid.
    If similarity < ``threshold``, the query is considered off-topic.

    Requires :func:`set_corpus_centroid` to have been called first.

    Args:
        query: The user's query string.
        threshold: Minimum cosine similarity to corpus centroid.

    Returns:
        ``(passed, reason)``
    """
    if _CORPUS_CENTROID is None:
        return True, ""  # Skip if centroid not set

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        q_emb = embedder.encode([query], show_progress_bar=False)[0]

        similarity = float(np.dot(q_emb, _CORPUS_CENTROID) /
                           (np.linalg.norm(q_emb) * np.linalg.norm(_CORPUS_CENTROID) + 1e-9))

        if similarity < threshold:
            logger.info("Input guardrail: embedding similarity %.3f < threshold %.3f",
                        similarity, threshold)
            return False, _REFUSAL

        return True, ""
    except Exception as exc:
        logger.warning("Embedding guardrail failed: %s", exc)
        return True, ""  # Fail-open


# =====================================================================
# Output groundedness verification
# =====================================================================


def _tokenize_simple(text: str) -> List[str]:
    """Lowercase whitespace tokeniser (works for EN + HI)."""
    return [t for t in text.lower().split() if len(t) > 1]


def _extract_entities(text: str) -> set:
    """Extract naive "entities" — capitalised words, numbers, proper nouns.

    This is a lightweight heuristic; a production system would use NER.
    """
    entities: set = set()
    # Capitalised multi-char words (likely proper nouns)
    for word in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
        entities.add(word.lower())
    # Numbers (years, measurements, etc.)
    for num in re.findall(r"\b\d+\.?\d*\b", text):
        entities.add(num)
    # Hindi words (Devanagari script tokens longer than 2 chars)
    for word in re.findall(r"[\u0900-\u097F]{3,}", text):
        entities.add(word)
    return entities


def verify_groundedness(
    answer: str,
    context_chunks: Sequence,
    token_threshold: float = _TOKEN_OVERLAP_THRESHOLD,
    entity_threshold: float = _ENTITY_OVERLAP_THRESHOLD,
) -> Tuple[bool, str]:
    """Verify that *answer* is grounded in *context_chunks*.

    Performs two checks:
        1. **Token overlap**: fraction of answer tokens that appear in the
           concatenated context.
        2. **Entity overlap**: fraction of answer "entities" (proper nouns,
           numbers) that appear in context.

    If either check fails, the answer is considered ungrounded and should
    be replaced with a refusal.

    Args:
        answer: The generated answer text.
        context_chunks: Retrieved chunks (any object with a ``.text`` attribute).
        token_threshold: Minimum token overlap ratio.
        entity_threshold: Minimum entity overlap ratio.

    Returns:
        ``(grounded, override_answer)`` — if ``grounded`` is ``False``,
        ``override_answer`` contains the refusal message.
    """
    if not answer or not context_chunks:
        return False, _REFUSAL

    # Build context text
    context_text = " ".join(getattr(c, "text", str(c)) for c in context_chunks)

    # ---- Token overlap ----
    answer_tokens = _tokenize_simple(answer)
    context_tokens_set = set(_tokenize_simple(context_text))

    if not answer_tokens:
        return False, _REFUSAL

    overlap_count = sum(1 for t in answer_tokens if t in context_tokens_set)
    token_ratio = overlap_count / len(answer_tokens)

    # ---- Entity overlap ----
    answer_entities = _extract_entities(answer)
    context_entities = _extract_entities(context_text)

    if answer_entities:
        entity_overlap = len(answer_entities & context_entities) / len(answer_entities)
    else:
        entity_overlap = 1.0  # No entities to check — pass

    logger.debug("Groundedness check: token_ratio=%.3f entity_overlap=%.3f",
                 token_ratio, entity_overlap)

    if token_ratio < token_threshold:
        return False, _REFUSAL
    if entity_overlap < entity_threshold:
        return False, _REFUSAL

    return True, ""
