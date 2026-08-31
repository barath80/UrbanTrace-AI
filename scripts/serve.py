"""Launch the UrbanTrace AI dashboard.

    python scripts/serve.py            # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Host to bind (0.0.0.0 allows phones on LAN to connect)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    import uvicorn
    print(f"UrbanTrace AI dashboard -> http://{a.host}:{a.port}")
    print(f"Local LAN / Phone Streamer access -> http://0.0.0.0:{a.port}")
    uvicorn.run("urbantrace.api.main:app", host=a.host, port=a.port,
                reload=a.reload, app_dir=str(ROOT / "backend"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
