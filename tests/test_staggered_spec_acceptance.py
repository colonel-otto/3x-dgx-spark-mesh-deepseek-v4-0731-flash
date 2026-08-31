#!/usr/bin/env python3
"""Tests for benchmark_staggered_spec_acceptance.py.

Validates prompt generation, metric scraping, tier sampling, and request
handling without requiring a live cluster.
"""

import importlib.util
import json
import math
import pathlib
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

# Load the module by file path (it lives in scripts/, not a package).
_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / \
    "benchmark_staggered_spec_acceptance.py"
spec = importlib.util.spec_from_file_location("staggered", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TestPromptConstruction(unittest.TestCase):
    def test_prompt_contains_salt(self):
        prompt = mod._build_prompt(2048, "test-salt-42")
        self.assertIn("test-salt-42", prompt)

    def test_prompt_length_scales_with_target(self):
        short = mod._build_prompt(1024, "short")
        long = mod._build_prompt(8192, "long")
        # Longer target -> longer string (approximate; not exact tokens)
        self.assertGreater(len(long), len(short))

    def test_prompt_is_deterministic(self):
        a = mod._build_prompt(4096, "same-salt")
        b = mod._build_prompt(4096, "same-salt")
        self.assertEqual(a, b)

    def test_different_salts_produce_different_prompts(self):
        a = mod._build_prompt(4096, "salt-a")
        b = mod._build_prompt(4096, "salt-b")
        self.assertNotEqual(a, b)


class TestTierSampling(unittest.TestCase):
    def test_samples_within_bounds(self):
        rng = random.Random(123)
        for _ in range(500):
            length = mod._sample_prompt_length(rng)
            self.assertGreaterEqual(length, 1024)
            self.assertLessEqual(length, 131072)

    def test_distribution_hits_all_tiers(self):
        rng = random.Random(456)
        samples = [mod._sample_prompt_length(rng) for _ in range(1000)]
        tier_a = [s for s in samples if 1024 <= s < 8192]
        tier_b = [s for s in samples if 8192 <= s < 32768]
        tier_c = [s for s in samples if 32768 <= s <= 131072]
        # All tiers should be represented
        self.assertGreater(len(tier_a), 0, "Tier A (1K-8K) not sampled")
        self.assertGreater(len(tier_b), 0, "Tier B (8K-32K) not sampled")
        self.assertGreater(len(tier_c), 0, "Tier C (32K-131K) not sampled")
        # Rough weight check: Tier A should be most frequent
        self.assertGreater(len(tier_a), len(tier_c))

    def test_deterministic_with_same_seed(self):
        a = [mod._sample_prompt_length(random.Random(99)) for _ in range(10)]
        b = [mod._sample_prompt_length(random.Random(99)) for _ in range(10)]
        self.assertEqual(a, b)


class TestMetricsScraper(unittest.TestCase):
    SAMPLE_METRICS = b"""\
# HELP vllm:spec_decode_num_accepted_tokens_total Total accepted
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total 12345.0
# HELP vllm:spec_decode_num_draft_tokens_total Total drafted
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total 18000.0
# HELP vllm:spec_decode_num_drafts_total Total draft iterations
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total 6000.0
# HELP vllm:num_preemptions_total Preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 0.0
# HELP vllm:num_requests_running Currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3.0
# HELP vllm:num_requests_waiting Currently waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1.0
"""

    def test_parses_all_counters(self):
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_METRICS
            mock_resp.__iter__ = lambda self: iter(self.read().split(b"\n"))
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            m = mod.scrape_metrics("http://fake/metrics")
            self.assertEqual(m["accepted"], 12345.0)
            self.assertEqual(m["draft"], 18000.0)
            self.assertEqual(m["draft_iters"], 6000.0)
            self.assertEqual(m["preemptions"], 0.0)
            self.assertEqual(m["running"], 3.0)
            self.assertEqual(m["waiting"], 1.0)

    def test_handles_missing_metrics_gracefully(self):
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"# empty\n"
            mock_resp.__iter__ = lambda self: iter(self.read().split(b"\n"))
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            m = mod.scrape_metrics("http://fake/metrics")
            self.assertEqual(m["accepted"], 0.0)
            self.assertEqual(m["preemptions"], 0.0)

    def test_handles_connection_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=Exception("Connection refused")):
            m = mod.scrape_metrics("http://unreachable/metrics")
            # Should return zeroes, not crash
            self.assertEqual(m["accepted"], 0.0)

    def test_ignores_created_sibling_metrics(self):
        """Counters like _created should not be picked up as the main counter."""
        data = (
            b"vllm:spec_decode_num_accepted_tokens_total 500.0\n"
            b"vllm:spec_decode_num_accepted_tokens_created 1693000000.0\n"
        )
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = data
            mock_resp.__iter__ = lambda self: iter(self.read().split(b"\n"))
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            m = mod.scrape_metrics("http://fake/metrics")
            self.assertEqual(m["accepted"], 500.0)


class TestRequestResultSchema(unittest.TestCase):
    """Verify the result dict schema from _send_request (mocked)."""

    def test_error_result_has_all_fields(self):
        import urllib.error as _ue
        with patch("urllib.request.urlopen",
                   side_effect=_ue.HTTPError(
                       "http://x", 500, "Internal Server Error", {}, None)):
            result = mod._send_request(
                "http://x/v1", "model", "prompt", 256, "req-1")
            self.assertIn("error", result)
            self.assertIn("HTTP 500", result["error"])
            self.assertEqual(result["request_id"], "req-1")
            self.assertFalse(result["window_ok"])


if __name__ == "__main__":
    unittest.main()
