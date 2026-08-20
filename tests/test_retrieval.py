import unittest
import time
from data.preprocess import Passage
from chunking.passage_native import PassageNativeChunker
from retrieval import build_retrieval_pipeline, retrieve, retrieve_with_scores

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        passages = [
            Passage(passage_id="p1", text="The capital of India is New Delhi located on Yamuna river.", language="eng", query_cluster="q1"),
            Passage(passage_id="p2", text="Mumbai is the financial capital of India.", language="eng", query_cluster="q1"),
            Passage(passage_id="p3", text="भारत की राजधानी नई दिल्ली है जो यमुना नदी के तट पर है।", language="hin", query_cluster="q2"),
            Passage(passage_id="p4", text="मुंबई भारत की आर्थिक राजधानी है।", language="hin", query_cluster="q2")
        ]
        self.chunks = PassageNativeChunker().chunk(passages)
        build_retrieval_pipeline(self.chunks)

    def test_retrieval_hybrid(self):
        res = retrieve("New Delhi capital", k=2)
        self.assertGreater(len(res), 0)

    def test_retrieval_language_filter(self):
        eng_res = retrieve("capital", k=5, language_filter="eng")
        self.assertTrue(all(c.language == "eng" for c in eng_res))
        
        hin_res = retrieve("राजधानी", k=5, language_filter="hin")
        self.assertTrue(all(c.language == "hin" for c in hin_res))

    def test_retrieval_latency(self):
        start = time.perf_counter()
        res = retrieve("financial capital", k=2)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        self.assertGreater(len(res), 0)
        self.assertLess(elapsed_ms, 200.0)

if __name__ == "__main__":
    unittest.main()
