"""Smoke probe for the running staging container.

After the Chainlit migration there is no gradio_client API to call, so this
script just verifies the server is reachable on the expected port and serving
HTTP. End-to-end RAG verification lives in scripts/smoke_e2e.py.
"""

import os
import sys

import httpx


def test_staging() -> None:
    target = os.getenv("SMOKE_TARGET_STAGING", "http://localhost:7860")
    print(f"[*] Probing Staging at {target} ...")
    try:
        resp = httpx.get(target, timeout=15.0, follow_redirects=True)
        if resp.status_code >= 500:
            print(f"[ERROR] Server returned {resp.status_code}")
            sys.exit(1)
        print(f"[SUCCESS] Server responded with {resp.status_code}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_staging()
