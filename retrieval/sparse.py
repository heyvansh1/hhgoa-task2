import re
import math
from typing import List, Tuple, Optional
from chunking.types import Chunk

class SparseRetriever:
    """
    In-process BM25 Sparse Keyword Retriever.
    Handles exact keyword matching for both English and Hindi text.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25 = None
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        # Multilingual word tokenization (extract English & Hindi word tokens)
        words = re.findall(r'\w+', text.lower())
        return words if words else text.lower().split()

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            return

        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception:
            self.bm25 = None

    def _fallback_bm25(self, query_tokens: List[str], k: int, candidates_indices: Optional[List[int]]) -> List[Tuple[Chunk, float]]:
        # Simple TF-IDF / term frequency overlap fallback
        scores = []
        indices = candidates_indices if candidates_indices is not None else range(len(self.chunks))
        
        for idx in indices:
            doc_tokens = self.tokenized_corpus[idx]
            if not doc_tokens:
                scores.append((idx, 0.0))
                continue
            doc_token_set = set(doc_tokens)
            score = sum(doc_tokens.count(t) for t in query_tokens if t in doc_token_set)
            scores.append((idx, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in scores[:k]]

    def search(self, query: str, k: int = 10, candidates_indices: Optional[List[int]] = None) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)

        if candidates_indices is not None and len(candidates_indices) == 0:
            return []

        if self.bm25 is not None:
            doc_scores = self.bm25.get_scores(query_tokens)
            if candidates_indices is not None:
                scored = [(idx, doc_scores[idx]) for idx in candidates_indices]
            else:
                scored = [(idx, doc_scores[idx]) for idx in range(len(self.chunks))]

            scored.sort(key=lambda x: x[1], reverse=True)
            return [(self.chunks[idx], float(score)) for idx, score in scored[:k]]
        else:
            return self._fallback_bm25(query_tokens, k, candidates_indices)
