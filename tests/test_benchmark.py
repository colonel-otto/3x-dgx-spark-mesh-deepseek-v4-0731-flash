import importlib.util
import pathlib
import sys
import unittest

P = pathlib.Path(__file__).parents[1] / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

class BenchmarkTests(unittest.TestCase):
    def test_prompt_contains_needle(self):
        p=m.make_prompt(2048,0.5)
        self.assertEqual(p.count("DGX_NEEDLE="), 2)
        self.assertIn(m.NEEDLE_VALUE,p)

    def test_position_changes_prompt(self):
        self.assertNotEqual(m.make_prompt(2048,0.1),m.make_prompt(2048,0.9))

if __name__ == "__main__": unittest.main()
