import pytest
from data.preprocess import Passage
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker

@pytest.fixture
def sample_passages():
    return [
        Passage(passage_id="p1", text="This is a short test passage.", language="eng", query_cluster="q1"),
        Passage(passage_id="p2", text=" ".join([f"word{i}" for i in range(100)]), language="eng", query_cluster="q1"),
        Passage(passage_id="p3", text="", language="hin", query_cluster="q2")
    ]

def test_passage_native_chunking(sample_passages):
    chunker = PassageNativeChunker()
    chunks = chunker.chunk(sample_passages)
    
    # 3 passages, so 3 chunks (even empty ones are currently kept by passage native, though preprocess normally filters them)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "pn_p1_0"
    assert chunks[0].source_passage_id == "p1"
    assert chunks[0].language == "eng"
    assert chunks[0].query_cluster == "q1"
    assert chunks[0].text == "This is a short test passage."

def test_fixed_size_chunking_short(sample_passages):
    chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
    chunks = chunker.chunk([sample_passages[0]])
    
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short test passage."

def test_fixed_size_chunking_long(sample_passages):
    chunker = FixedSizeChunker(chunk_size=30, overlap_percent=0.2)
    # overlap = 6. step = 24. 
    # 100 words -> chunk 0 (0-30), chunk 1 (24-54), chunk 2 (48-78), chunk 3 (72-100), chunk 4 (96-100) -> Actually 100 words means len is 100.
    chunks = chunker.chunk([sample_passages[1]])
    
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "fs_p2_0"
    assert len(chunks[0].text.split()) == 30
    assert len(chunks[1].text.split()) == 30
    
def test_fixed_size_chunking_empty(sample_passages):
    chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
    chunks = chunker.chunk([sample_passages[2]])
    assert len(chunks) == 0  # Should avoid empty chunks
