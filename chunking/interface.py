from typing import List, Protocol
from data.preprocess import Passage
from chunking.types import Chunk

class Chunker(Protocol):
    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Convert a list of Passages into a list of Chunks."""
        ...
