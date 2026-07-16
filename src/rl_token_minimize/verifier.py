import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT = 2000


@dataclass
class VerifierResult:
    passed: bool
    output: str


def run_tests(workspace: Path, test_code: str, timeout: int = 10) -> VerifierResult:
    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "hidden_test.py"
        test_file.write_text(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(passed=False, output=f"timeout after {timeout}s")
        output = (proc.stdout + proc.stderr).strip()[-MAX_OUTPUT:]
        return VerifierResult(passed=proc.returncode == 0, output=output)
