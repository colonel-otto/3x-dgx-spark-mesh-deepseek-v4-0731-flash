#!/usr/bin/env python3
"""Fail the test suite if sensitive data is tracked in the repo.

This is a public repo. Serial numbers, real email addresses, personal names,
and home directories that leak a username must never be committed. The scanner
lives in scripts/check_no_sensitive.py; this wires it into `make test` so CI
catches a leak before it is published rather than after.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER = os.path.join(REPO, "scripts", "check_no_sensitive.py")


class TestNoSensitiveData(unittest.TestCase):
    def run_scanner(self, *args):
        return subprocess.run([sys.executable, SCANNER, *args],
                              capture_output=True, text=True, cwd=REPO)

    def test_repo_is_clean(self):
        result = subprocess.run([sys.executable, SCANNER],
                                capture_output=True, text=True, cwd=REPO)
        self.assertEqual(
            result.returncode, 0,
            "sensitive data found in tracked files:\n" + result.stdout)

    def test_scanner_actually_detects(self):
        """A scanner that never fires is worse than none - prove it fires."""
        probe = os.path.join(REPO, "_sensitive_probe.tmp")
        # Assembled at runtime so this file holds no serial-shaped literal for
        # the scanner to flag when it inevitably scans itself.
        planted = "DGX_SERIAL" + "_NUMBER=" + chr(34) + "AB1234567890" + chr(34)
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(planted + "\n")
        try:
            subprocess.run(["git", "-C", REPO, "add", "-N", probe],
                           capture_output=True)
            result = subprocess.run([sys.executable, SCANNER],
                                    capture_output=True, text=True, cwd=REPO)
            self.assertEqual(result.returncode, 1,
                             "scanner did not flag a planted serial number")
            self.assertIn("hardware serial", result.stdout)
        finally:
            subprocess.run(["git", "-C", REPO, "rm", "-q", "-f", "--cached", probe],
                           capture_output=True)
            if os.path.exists(probe):
                os.remove(probe)

    def test_allowed_email_does_not_exempt_real_email_on_same_line(self):
        probe = os.path.join(REPO, "_sensitive_email_probe.tmp")
        # Assemble the private address so the repository-wide scanner does not
        # flag this test's own source.
        leaked = "person" + chr(64) + "private.invalid"
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("safe@example.com and " + leaked + "\n")
        try:
            subprocess.run(["git", "-C", REPO, "add", "-N", probe],
                           capture_output=True)
            result = self.run_scanner()
            self.assertEqual(result.returncode, 1)
            self.assertIn("real email address", result.stdout)
        finally:
            subprocess.run(["git", "-C", REPO, "rm", "-q", "-f", "--cached", probe],
                           capture_output=True)
            if os.path.exists(probe):
                os.remove(probe)

    def test_staged_scan_reads_index_blob(self):
        probe = os.path.join(REPO, "_sensitive_staged_probe.tmp")
        leaked = "person" + chr(64) + "private.invalid"
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(leaked + "\n")
        try:
            subprocess.run(["git", "-C", REPO, "add", probe], check=True,
                           capture_output=True)
            # A safe working-tree replacement must not hide the staged leak.
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("REDACTED\n")
            result = self.run_scanner("--staged")
            self.assertEqual(result.returncode, 1)
            self.assertIn("real email address", result.stdout)
        finally:
            subprocess.run(["git", "-C", REPO, "rm", "-q", "-f", "--cached", probe],
                           capture_output=True)
            if os.path.exists(probe):
                os.remove(probe)

    def test_scanner_detects_bare_username_assignment(self):
        probe = os.path.join(REPO, "_sensitive_username_probe.tmp")
        # Assemble the label so this test does not flag its own source.
        label = "USER" + "NAME"
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(label + "=privateoperator\n")
        try:
            subprocess.run(["git", "-C", REPO, "add", "-N", probe],
                           capture_output=True)
            result = self.run_scanner()
            self.assertEqual(result.returncode, 1)
            self.assertIn("username assignment", result.stdout)
        finally:
            subprocess.run(["git", "-C", REPO, "rm", "-q", "-f", "--cached", probe],
                           capture_output=True)
            if os.path.exists(probe):
                os.remove(probe)


if __name__ == "__main__":
    unittest.main()
