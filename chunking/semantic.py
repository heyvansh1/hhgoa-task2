import re
import numpy as np
from typing import List, Optional
from data.preprocess import Passage
from chunking.types import Chunk
from chunking.interface import Chunker

class SemanticChunker(Chunker):
    """
    Semantic Chunker:
    Splits text into sentences, computes sentence embeddings,
    and places chunk boundaries at points of high cosine distance.
    """
    def __init__(self, distance_threshold: float = 0.4, max_chunk_words: int = 150):
        self.distance_threshold = distance_threshold
        self.max_chunk_words = max_chunk_words
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder = "fallback"
        return self._embedder

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries (English .!? and Hindi danda ।)
        raw_sentences = re.split(r'(?<=[.!?।])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(1.0 - similarity)

    def _compute_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        embedder = self._get_embedder()
        if embedder != "fallback":
            try:
                embeddings = embedder.encode(sentences, show_progress_bar=False)
                return np.array(embeddings)
            except Exception:
                pass
        # Fallback simple TF-IDF / character n-gram pseudo-embedding if transformer unavailable
        vecs = []
        for s in sentences:
            words = s.lower().split()
            # Simple 128-dim bag of words hashing trick vector
            v = np.zeros(128, dtype=np.float32)
            for w in words:
                v[hash(w) % 128] += 1.0
            vecs.append(v)
        return np.array(vecs)

    def chunk_passage(self, passage: Passage) -> List[Chunk]:
        sentences = self._split_sentences(passage.text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    chunk_id=f"sem_{passage.passage_id}_0",
                    source_passage_id=passage.passage_id,
                    language=passage.language,
                    text=passage.text,
                    query_cluster=passage.query_cluster,
                    metadata={"strategy": "semantic", "num_sentences": len(sentences)}
                )
            ]

        embeddings = self._compute_embeddings(sentences)
        distances = []
        for i in range(len(sentences) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)

        # Form chunks by grouping sentences until distance exceeds threshold or max words reached
        chunks: List[Chunk] = []
        curr_sentences: List[str] = [sentences[0]]
        curr_words = len(sentences[0].split())
        chunk_idx = 0

        for i in range(len(distances)):
            dist = distances[i]
            next_sent = sentences[i+1]
            next_words = len(next_sent.split())

            if dist > self.distance_threshold or (curr_words + next_words > self.max_chunk_words):
                # Cut chunk
                chunk_text = " ".join(curr_sentences)
                chunks.append(
                    Chunk(
                        chunk_id=f"sem_{passage.passage_id}_{chunk_idx}",
                        source_passage_id=passage.passage_id,
                        language=passage.language,
                        text=chunk_text,
                        query_cluster=passage.query_cluster,
                        metadata={"strategy": "semantic", "break_distance": round(dist, 4)}
                    )
                )
                chunk_idx += 1
                curr_sentences = [next_sent]
                curr_words = next_words
            else:
                curr_sentences.append(next_sent)
                curr_words += next_words

        if curr_sentences:
            chunk_text = " ".join(curr_sentences)
            chunks.append(
                Chunk(
                    chunk_id=f"sem_{passage.passage_id}_{chunk_idx}",
                    source_passage_id=passage.passage_id,
                    language=passage.language,
                    text=chunk_text,
                    query_cluster=passage.query_cluster,
                    metadata={"strategy": "semantic"}
                )
            )

        return chunks

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        all_chunks = []
        for passage in passages:
            all_chunks.extend(self.chunk_passage(passage))
        return all_chunks
