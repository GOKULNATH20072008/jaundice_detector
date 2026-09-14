"""Periodic real-API validation of the eye gate.

This calls the live Anthropic Messages API for every fixture image and prints
pass/fail. Requires ANTHROPIC_API_KEY in the environment (do not hardcode).
Run:  python tests/run_integration.py

Exit code 0 = all expected verdicts matched; 1 = any mismatch / missing key.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import eye_gate  # noqa: E402
from test_eye_gate import CASES, FIXTURES, load_fixture  # noqa: E402


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — integration test requires a live key.")
        return 1

    failures = 0
    for fixture, expected, note in CASES:
        image = load_fixture(fixture)
        result = eye_gate.gate_is_real_eye(image)
        ok = result["is_real_human_eye"] == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status:4}] {fixture:<22} expected={str(expected):5} got={result}")
        failures += 0 if ok else 1

    print("\n", "All gate fixtures matched." if failures == 0 else f"{failures} mismatch(es).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())