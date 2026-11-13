"""Bridges the Foundry engine self-test (Node) into the Python CI suite.

The Universal Builder engine is JS (it runs in the browser), but it is pure and
headless-runnable, so we execute its self-test under Node and assert it passes —
proving the evolutionary search writes & runs real code and improves fitness.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SELFTEST = _ROOT / "foundry" / "selftest.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_foundry_engine_selftest_passes():
    assert _SELFTEST.exists(), "foundry/selftest.js missing"
    proc = subprocess.run(["node", str(_SELFTEST)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"foundry self-test failed:\n{proc.stdout}\n{proc.stderr}"
    assert "ALL PASSED" in proc.stdout
