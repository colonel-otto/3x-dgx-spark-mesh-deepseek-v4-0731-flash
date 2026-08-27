"""Keep the human and machine-readable result indexes complete."""

import glob
import importlib.util
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_index():
    return yaml.safe_load((RESULTS / "index.yaml").read_text(encoding="utf-8"))


class ResultsIndexTests(unittest.TestCase):
    def test_every_result_directory_is_indexed_once(self):
        directories = {path.name for path in RESULTS.iterdir() if path.is_dir()}
        yaml_text = (RESULTS / "index.yaml").read_text(encoding="utf-8")
        indexed = re.findall(r"^  - dir: (\S+)\s*$", yaml_text, re.MULTILINE)

        self.assertEqual(len(indexed), len(set(indexed)), "duplicate index.yaml entry")
        self.assertEqual(directories, set(indexed))

        total = re.search(r"^    total: (\d+)\s*$", yaml_text, re.MULTILINE)
        self.assertIsNotNone(total, "index.yaml has no fabric-gate total")
        self.assertEqual(int(total.group(1)), len(directories))

    def test_every_result_directory_is_in_human_catalogue(self):
        directories = {path.name for path in RESULTS.iterdir() if path.is_dir()}
        catalogue = (RESULTS / "README.md").read_text(encoding="utf-8")
        missing = sorted(
            name for name in directories if f"]({name}/)" not in catalogue
        )
        self.assertEqual([], missing)

    def test_statuses_references_and_raw_artifacts_are_valid(self):
        data = load_index()
        entries = data["directories"]
        names = {entry["dir"] for entry in entries}
        allowed_gates = set(data["meta"]["fabric_gate_values"])

        for entry in entries:
            with self.subTest(directory=entry["dir"]):
                status = str(entry["status"])
                self.assertTrue(
                    status == "CURRENT"
                    or status.startswith("VOID-")
                    or status.startswith("SUPERSEDED-BY-"),
                    f"invalid status: {status}",
                )
                self.assertIn(entry["fabric_gate"], allowed_gates)

                replacement = entry.get("superseded_by")
                if replacement:
                    self.assertIn(replacement, names)

                if entry.get("raw_data_committed"):
                    self.assertTrue(entry.get("raw_files"), "raw files not listed")
                    for pattern in entry["raw_files"]:
                        matches = glob.glob(
                            str(RESULTS / entry["dir"] / str(pattern))
                        )
                        self.assertTrue(matches, f"missing raw artifact: {pattern}")

    def test_readme_status_matches_index_when_readme_exists(self):
        for entry in load_index()["directories"]:
            readme = RESULTS / entry["dir"] / "README.md"
            if not readme.exists():
                continue
            text = readme.read_text(encoding="utf-8", errors="replace")
            self.assertIn(
                str(entry["status"]),
                text,
                f"{entry['dir']} README lacks its authoritative status",
            )

    def test_generated_markdown_is_current(self):
        script = ROOT / "scripts" / "generate_results_index.py"
        spec = importlib.util.spec_from_file_location("generate_results_index", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        expected = module.render(load_index())
        actual = (RESULTS / "INDEX.md").read_text(encoding="utf-8")
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
