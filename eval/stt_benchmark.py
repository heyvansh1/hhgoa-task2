import json
import os
from typing import List, Dict

from stt.sarvam import SarvamSTT
from stt.elevenlabs import ElevenLabsSTT

def load_manifest(manifest_path: str) -> List[Dict]:
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_benchmark():
    manifest_path = "data/samples/manifest.json"
    samples = load_manifest(manifest_path)
    
    if not samples:
        print("No samples to benchmark.")
        return

    providers = {
        "sarvam": SarvamSTT(),
        "elevenlabs": ElevenLabsSTT()
    }
    
    results = []
    
    for sample in samples:
        audio_path = sample["file_path"]
        if not os.path.exists(audio_path):
            print(f"Audio file missing, skipping: {audio_path}")
            continue
            
        for name, provider in providers.items():
            print(f"Testing {name} on {sample['sample_id']}...")
            res = provider.transcribe(audio_path)
            results.append({
                "sample_id": sample["sample_id"],
                "language": sample["language"],
                "provider": name,
                "transcript": res.transcript,
                "latency_ms": res.latency_ms,
                "error": res.error,
                "reference": sample["reference"]
            })
            
    print("\n--- Benchmark Results ---")
    for r in results:
        print(r)

if __name__ == "__main__":
    run_benchmark()
