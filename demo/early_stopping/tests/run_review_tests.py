"""Run the isolated backend suite and record its exact reviewed source bytes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parents[2]
command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "demo/early_stopping/tests", "-p", "test_*.py", "-v"]
environment = os.environ.copy()
environment["PYTHONDONTWRITEBYTECODE"] = "1"
started = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(TESTS / "backend_test.log").write_text(result.stdout, encoding="utf-8")
paths = [ROOT / "demo/early_stopping" / name for name in ("runtime.py", "server.py", "streaming_generator.py")]
paths += sorted(TESTS.glob("*.py"))
record = {
    "command": command, "cwd": str(ROOT), "exit_code": result.returncode,
    "elapsed_seconds": time.perf_counter() - started,
    "scope": "Injected deterministic generator and predictor; no language-model generation, fitting, training, regrading or dependency changes.",
    "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    "log": "demo/early_stopping/tests/backend_test.log",
}
(TESTS / "backend_test_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(result.stdout, end="")
raise SystemExit(result.returncode)
