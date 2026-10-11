"""Persist actual unittest evidence; never import historical runners/models."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

phase = Path(__file__).resolve().parents[1]
result = subprocess.run(
    [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
    cwd=phase, capture_output=True, text=True, encoding="utf-8", errors="replace")
test = phase / "tests/test_readonly_firewall.py"
audit = phase / "scripts/audit_k_readonly.py"
record = {
    "command": [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "returncode": result.returncode,
    "production_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    "test_module_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    "stdout": result.stdout, "stderr": result.stderr,
    "scope": "synthetic access-control checks only; no model or scientific metrics",
}
out = phase / "outputs/K_FIREWALL_TEST_RESULTS.json"
out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(result.stdout, end="")
print(result.stderr, end="")
raise SystemExit(result.returncode)
