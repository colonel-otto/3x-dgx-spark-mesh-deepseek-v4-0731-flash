import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.exclusivity as exclusivity


class ExclusivityTests(unittest.TestCase):

    def test_parse_metrics(self):
        sample_metrics = """
# HELP vllm:num_requests_running Number of requests
vllm:num_requests_running{engine="0"} 0.0
# HELP vllm:num_requests_waiting Number of waiting
vllm:num_requests_waiting 0.0
# HELP vllm:request_success_total Number of success
vllm:request_success_total 1234.0
"""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = sample_metrics.encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp

            m = exclusivity.get_engine_metrics("http://fake/metrics")
            self.assertEqual(m["vllm:num_requests_running"], 0.0)
            self.assertEqual(m["vllm:num_requests_waiting"], 0.0)
            self.assertEqual(m["vllm:request_success_total"], 1234.0)

    def test_assert_idle_success(self):
        with patch("scripts.exclusivity.get_engine_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "vllm:num_requests_running": 0.0,
                "vllm:num_requests_waiting": 0.0,
                "vllm:request_success_total": 500.0,
            }
            start_val = exclusivity.assert_idle(timeout_s=1.0)
            self.assertEqual(start_val, 500.0)

    def test_assert_idle_timeout(self):
        with patch("scripts.exclusivity.get_engine_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "vllm:num_requests_running": 2.0,
                "vllm:num_requests_waiting": 0.0,
                "vllm:request_success_total": 500.0,
            }
            with self.assertRaises(RuntimeError):
                exclusivity.assert_idle(timeout_s=0.1, poll_interval_s=0.05)

    def test_verify_exclusivity_pass(self):
        with patch("scripts.exclusivity.get_engine_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "vllm:request_success_total": 550.0,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / "exclusivity.json"
                rec = exclusivity.verify_exclusivity(
                    start_success_total=500.0,
                    expected_requests=50,
                    output_file=out_path,
                )
                self.assertEqual(rec["status"], "PASS")
                self.assertEqual(rec["actual_requests_delta"], 50)
                self.assertTrue(rec["is_exclusive"])
                self.assertTrue(out_path.exists())

    def test_verify_exclusivity_fail(self):
        with patch("scripts.exclusivity.get_engine_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "vllm:request_success_total": 555.0,  # 55 delta instead of 50
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / "exclusivity.json"
                with self.assertRaises(RuntimeError):
                    exclusivity.verify_exclusivity(
                        start_success_total=500.0,
                        expected_requests=50,
                        output_file=out_path,
                    )


if __name__ == "__main__":
    unittest.main()
