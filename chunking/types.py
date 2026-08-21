"""Core data types for the chunking subsystem."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """A retrieval unit produced by any chunking strategy.

    Attributes:
        chunk_id: Globally unique identifier for this chunk.
        text: The textual content of this chunk.
        source_passage_id: ID of the parent passage this chunk was derived from.
        language: ISO-639-3 language code (e.g. "eng", "hin").
        metadata: Strategy-specific metadata (parent_id, hierarchy_level,
                  embedding_model, split_threshold, char_length, etc.).
        query_cluster: Optional query cluster ID from the source dataset.
    """

    chunk_id: str
    text: str
    source_passage_id: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_cluster: Optional[str] = None
