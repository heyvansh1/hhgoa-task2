import unittest
from data.preprocess import Passage
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sample_passages = [
            Passage(passage_id="p1", text="This is a short test passage.", language="eng", query_cluster="q1"),
            Passage(passage_id="p2", text=" ".join([f"word{i}" for i in range(100)]), language="eng", query_cluster="q1"),
            Passage(passage_id="p3", text="", language="hin", query_cluster="q2")
        ]

    def test_passage_native_chunking(self):
        chunker = PassageNativeChunker()
        chunks = chunker.chunk(self.sample_passages)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "pn_p1_0")
        self.assertEqual(chunks[0].source_passage_id, "p1")
        self.assertEqual(chunks[0].language, "eng")
        self.assertEqual(chunks[0].query_cluster, "q1")
        self.assertEqual(chunks[0].text, "This is a short test passage.")

    def test_fixed_size_chunking_short(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
        chunks = chunker.chunk([self.sample_passages[0]])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "This is a short test passage.")

    def test_fixed_size_chunking_long(self):
        chunker = FixedSizeChunker(chunk_size=30, overlap_percent=0.2)
        chunks = chunker.chunk([self.sample_passages[1]])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "fs_p2_0")
        self.assertEqual(len(chunks[0].text.split()), 30)
        self.assertEqual(len(chunks[1].text.split()), 30)

    def test_fixed_size_chunking_empty(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap_percent=0.2)
        chunks = chunker.chunk([self.sample_passages[2]])
        self.assertEqual(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()
