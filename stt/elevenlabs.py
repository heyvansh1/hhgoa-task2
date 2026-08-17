import os
import time
import requests
from typing import Optional
from .base import BaseSTT, STTResult

class ElevenLabsSTT(BaseSTT):
    def __init__(self):
        super().__init__("elevenlabs")
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.url = "https://api.elevenlabs.io/v1/speech-to-text"

    def transcribe(self, audio_path: str) -> STTResult:
        if not self.api_key:
            return STTResult(transcript="", latency_ms=0, provider=self.provider_name, error="ELEVENLABS_API_KEY not set")
            
        start_time = time.time()
        
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                headers = {"xi-api-key": self.api_key}
                data = {
                    "model_id": "scribe_v1"
                }
                
                response = requests.post(self.url, headers=headers, files=files, data=data)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("text", "")
                    return STTResult(transcript=transcript, latency_ms=latency, provider=self.provider_name)
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    return STTResult(transcript="", latency_ms=latency, provider=self.provider_name, error=error_msg)
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return STTResult(transcript="", latency_ms=latency, provider=self.provider_name, error=str(e))
