import os
import time
import requests
from typing import Optional
from .base import BaseSTT, STTResult

class SarvamSTT(BaseSTT):
    def __init__(self):
        super().__init__("sarvam")
        self.api_key = os.environ.get("SARVAM_API_KEY")
        self.url = "https://api.sarvam.ai/speech-to-text-translate" # Saarika API endpoint

    def transcribe(self, audio_path: str) -> STTResult:
        if not self.api_key:
            return STTResult(transcript="", latency_ms=0, provider=self.provider_name, error="SARVAM_API_KEY not set")
        
        start_time = time.time()
        
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                # Note: Assuming 'hi-IN' or auto-detect is passed as needed in a real application
                # For Saarika, we usually translate or transcribe. Let's assume a generic call structure.
                payload = {"prompt": ""}
                headers = {"api-subscription-key": self.api_key}
                
                response = requests.post(self.url, data=payload, files=files, headers=headers)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("transcript", "")
                    # Sarvam Saarika might return different structure depending on exact endpoint used, 
                    # but typically it's {"transcript": "..."}
                    return STTResult(transcript=transcript, latency_ms=latency, provider=self.provider_name)
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    return STTResult(transcript="", latency_ms=latency, provider=self.provider_name, error=error_msg)
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return STTResult(transcript="", latency_ms=latency, provider=self.provider_name, error=str(e))
