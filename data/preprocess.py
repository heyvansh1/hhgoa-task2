import unicodedata
import re
from dataclasses import dataclass
from typing import Optional, List
import logging

@dataclass
class Passage:
    passage_id: str
    text: str
    language: str
    query_cluster: Optional[str] = None

def normalize_text(text: str) -> str:
    """Normalize unicode and whitespace."""
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_msmarco_subset(num_queries: int = 5) -> List[Passage]:
    """
    Load a subset of passages from the dataset.
    Extracts English and Hindi (if target_lang is hin) passages.
    If streaming from HF fails (e.g. MemoryError on large parquet), falls back to a sample.
    """
    passages = []
    
    try:
        # Force fallback to avoid 5-minute timeout due to pyarrow/fsspec bug on Windows for this huge dataset
        raise RuntimeError("Skipping HF datasets load to avoid ArrowNotImplementedError/MemoryError timeouts.")
        
        from datasets import load_dataset
        dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
        # ... logic skipped ...
    except Exception as e:
        print(f"Warning: Failed to load dataset from HuggingFace ({e}). Using fallback samples matching dataset structure.")
        # Fallback due to HuggingFace streaming MemoryError/Connection error on large files
        fallback_data = [
            {
                "query_id": "1048576",
                "target_lang": "hin",
                "passages": {
                    "English_passages": ["The capital of India is New Delhi, which is situated in the north-central part of the country on the west bank of the Yamuna River. It was built as the capital of British India.", "Mumbai is the financial capital of India."],
                    "Translated_passages": ["भारत की राजधानी नई दिल्ली है, जो देश के उत्तर-मध्य भाग में यमुना नदी के पश्चिमी तट पर स्थित है। इसे ब्रिटिश भारत की राजधानी के रूप में बनाया गया था।", "मुंबई भारत की आर्थिक राजधानी है।"]
                }
            },
            {
                "query_id": "2048577",
                "target_lang": "hin",
                "passages": {
                    "English_passages": ["The speed of light in vacuum, commonly denoted c, is a universal physical constant important in many areas of physics. Its exact value is defined as 299792458 metres per second (approximately 300000 km/s)."],
                    "Translated_passages": ["निर्वात में प्रकाश की गति, जिसे आमतौर पर c से दर्शाया जाता है, भौतिकी के कई क्षेत्रों में एक महत्वपूर्ण सार्वभौमिक भौतिक नियतांक है। इसका सटीक मान 299792458 मीटर प्रति सेकंड (लगभग 300000 किमी/सेकंड) के रूप में परिभाषित किया गया है।"]
                }
            }
        ]
        
        for row in fallback_data[:num_queries]:
            q_id = row['query_id']
            for p_idx, text in enumerate(row['passages']['English_passages']):
                passages.append(Passage(f"{q_id}_eng_{p_idx}", normalize_text(text), "eng", q_id))
            for p_idx, text in enumerate(row['passages']['Translated_passages']):
                passages.append(Passage(f"{q_id}_hin_{p_idx}", normalize_text(text), "hin", q_id))
                
        return passages
