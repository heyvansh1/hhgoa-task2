import numpy as np
from typing import List, Tuple, Optional
from chunking.types import Chunk

class DenseRetriever:
    """
    In-process FAISS/Vector Dense Retriever.
    Embeds chunk texts and performs fast vector similarity search using inner product / cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedder = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
            except Exception:
                self._embedder = "fallback"
        return self._embedder

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embedder = self._get_embedder()
        if embedder != "fallback":
            try:
                vecs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                return np.array(vecs, dtype=np.float32)
            except Exception:
                pass
        
        # Fast fallback: 128-dim normalized character n-gram + word hashing vectors
        vecs = []
        for t in texts:
            words = t.lower().split()
            v = np.zeros(128, dtype=np.float32)
            for w in words:
                v[hash(w) % 128] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [c.text for c in chunks]
        self.embeddings = self._embed_texts(texts)

        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
        except Exception:
            self.faiss_index = None

    def search(self, query: str, k: int = 10, candidates_indices: Optional[List[int]] = None) -> List[Tuple[Chunk, float]]:
        if not self.chunks or self.embeddings is None:
            return []

        query_vec = self._embed_texts([query])[0]

        if candidates_indices is not None and len(candidates_indices) == 0:
            return []

        if candidates_indices is not None:
            # Filter search among allowed candidates
            cand_embeddings = self.embeddings[candidates_indices]
            scores = np.dot(cand_embeddings, query_vec)
            top_k_sub = np.argsort(scores)[::-1][:k]
            results = []
            for sub_idx in top_k_sub:
                orig_idx = candidates_indices[sub_idx]
                results.append((self.chunks[orig_idx], float(scores[sub_idx])))
            return results

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(np.array([query_vec]), min(k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            scores = np.dot(self.embeddings, query_vec)
            top_k_idx = np.argsort(scores)[::-1][:k]
            return [(self.chunks[i], float(scores[i])) for i in top_k_idx]
