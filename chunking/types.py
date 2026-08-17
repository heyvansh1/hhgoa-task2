from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    chunk_id: str
    source_passage_id: str
    language: str
    text: str
    query_cluster: Optional[str] = None
