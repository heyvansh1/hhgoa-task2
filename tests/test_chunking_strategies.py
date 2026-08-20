import unittest
from data.preprocess import Passage
from chunking_strategies import get_chunker, STRATEGIES, chunk_with_strategy

class TestChunkingStrategies(unittest.TestCase):
    def setUp(self):
        self.sample_passages = [
            Passage(passage_id="p1", text="The capital of India is New Delhi. It is a historic city.", language="eng", query_cluster="q1"),
            Passage(passage_id="p2", text="भारत की राजधानी नई दिल्ली है। यह एक ऐतिहासिक शहर है।", language="hin", query_cluster="q1")
        ]

    def test_all_strategies_instantiation(self):
        for name in STRATEGIES:
            chunker = get_chunker(name)
            self.assertIsNotNone(chunker)

    def test_semantic_chunker(self):
        chunker = get_chunker("semantic", distance_threshold=0.3)
        chunks = chunker.chunk(self.sample_passages)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c.metadata.get("strategy") == "semantic" for c in chunks))

    def test_metadata_aware_chunker(self):
        chunks = chunk_with_strategy(self.sample_passages, "metadata_aware")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue("[Lang: eng" in chunks[0].text or "[Lang: hin" in chunks[1].text)
        self.assertEqual(chunks[0].metadata.get("strategy"), "metadata_aware")

    def test_hierarchical_chunker(self):
        chunker = get_chunker("hierarchical", child_chunk_size=10)
        chunks = chunker.chunk(self.sample_passages)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIsNotNone(chunks[0].parent_chunk_id)
        self.assertEqual(chunks[0].parent_text, self.sample_passages[0].text)
        self.assertEqual(chunks[0].metadata.get("strategy"), "hierarchical")

if __name__ == "__main__":
    unittest.main()
