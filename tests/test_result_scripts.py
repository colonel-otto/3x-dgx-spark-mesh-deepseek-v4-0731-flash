from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResultScriptSyntaxTests(unittest.TestCase):
    def test_retained_result_scripts_parse_as_bash(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")

        scripts = sorted((ROOT / "results").glob("**/*.sh"))
        self.assertTrue(scripts, "expected at least one retained result script")

        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                subprocess.run(
                    [bash, "-n", str(script)],
                    check=True,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
