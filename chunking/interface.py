"""Abstract base class for all chunking strategies."""

from abc import ABC, abstractmethod
from typing import List

from data.preprocess import Passage
from chunking.types import Chunk


class BaseChunker(ABC):
    """Abstract base for every chunking strategy.

    Subclasses must implement ``chunk`` which converts a list of raw
    :class:`Passage` objects into a list of retrieval-ready :class:`Chunk`
    objects.
    """

    @abstractmethod
    def chunk(self, passages: List[Passage]) -> List[Chunk]:
        """Convert a list of Passages into a list of Chunks.

        Args:
            passages: Source passages from the MSMARCO-XI dataset.

        Returns:
            A list of :class:`Chunk` objects ready for indexing.
        """
        ...

    @property
    def strategy_name(self) -> str:
        """Human-readable name of the chunking strategy."""
        return self.__class__.__name__
