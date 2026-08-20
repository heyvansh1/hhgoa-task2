from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    source_passage_id: str
    language: str
    text: str
    query_cluster: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
