from typing import List
from data.preprocess import Passage
from chunking.types import Chunk

class FixedSizeChunker:
    """
    Baseline #2: Fixed-size + overlap chunking.
    Uses a simple whitespace tokenizer (words = tokens) to keep dependencies lightweight.
    Target size: 256 tokens, Overlap: 20%.
    """
    def __init__(self, chunk_size: int = 256, overlap_percent: float = 0.20):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_percent)
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer suitable for English and Hindi text baselines.
        return text.split()
        
    def _detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        chunks = []
        for p in passages:
            tokens = self._tokenize(p.text)
            if not tokens:
                continue
                
            step = self.chunk_size - self.overlap
            if step <= 0:
                step = 1 # Fallback safeguard
                
            chunk_idx = 0
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self._detokenize(chunk_tokens)
                
                if chunk_text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"fs_{p.passage_id}_{chunk_idx}",
                        source_passage_id=p.passage_id,
                        language=p.language,
                        text=chunk_text,
                        query_cluster=p.query_cluster
                    ))
                    chunk_idx += 1
                    
                # Break if this chunk reached the end of the text
                if i + self.chunk_size >= len(tokens):
                    break
                    
        return chunks
