import importlib.util
import pathlib
import unittest

P = pathlib.Path(__file__).parents[1] / "scripts" / "compare_results.py"
spec = importlib.util.spec_from_file_location("compare_results", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class CompareTests(unittest.TestCase):
    def test_pct(self):
        self.assertEqual(m.pct(10, 12), 20)
        self.assertEqual(m.pct(10, 8), -20)
        self.assertIsNone(m.pct(0, 8))

    def test_summary(self):
        rows=[
            {"context_target":2048,"concurrency":1,"ok":True,"needle_correct":True,"ttft_s":1.0,"e2e_s":3.0,"decode_tps":20.0,"wave_aggregate_output_tps":18.0,"prompt_tokens":2000},
            {"context_target":2048,"concurrency":1,"ok":True,"needle_correct":False,"ttft_s":2.0,"e2e_s":4.0,"decode_tps":22.0,"wave_aggregate_output_tps":19.0,"prompt_tokens":2010},
        ]
        s=m.summarize(rows)[(2048,1)]
        self.assertEqual(s["requests"],2)
        self.assertEqual(s["needle_rate"],0.5)
        self.assertEqual(s["ttft"],1.5)
        self.assertEqual(s["decode_tps"],21.0)

if __name__ == "__main__": unittest.main()
