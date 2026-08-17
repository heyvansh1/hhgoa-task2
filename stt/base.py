from dataclasses import dataclass
from typing import Optional

@dataclass
class STTResult:
    transcript: str
    latency_ms: float
    provider: str
    language: Optional[str] = None
    error: Optional[str] = None

class BaseSTT:
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def transcribe(self, audio_path: str) -> STTResult:
        raise NotImplementedError("Subclasses must implement transcribe()")
