"""One command to run UrbanTrace AI end to end.

    python run.py            # process the videos, then serve the dashboard
    python run.py serve      # just serve (skip processing)
    python run.py process    # just process + persist, no server
    python run.py check      # prerequisites and self-test only
    python run.py build-web  # rebuild the React frontend

Each step is skipped with a clear message rather than failing when its inputs
are missing, so a partial setup still gets you as far as it can.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

PY = sys.executable
LINE = "=" * 74


def head(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def run(cmd: list, cwd: Path | None = None) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.call(cmd, cwd=str(cwd or ROOT))


# ---------------------------------------------------------------- checks

def check() -> bool:
    """Report what is present. Returns False only for a fatal gap."""
    head("PREREQUISITES")
    ok = True

    try:
        from urbantrace.core.runtime import describe
        rt = describe()
        print(f"  torch      : {rt['torch']}  device={rt['device']}"
              + (f"  ({rt['gpu_name']})" if rt.get("gpu_name") else ""))
        if not rt.get("cuda"):
            print("               CPU only - processing will be slow but correct.")
    except Exception as e:
        print(f"  torch      : MISSING ({type(e).__name__}: {e})")
        print("               pip install -r requirements.txt")
        ok = False

    weights = ROOT / "models" / "weights"
    need = ["yolo11m.pt", "license-plate-finetune-v1m.pt"]
    missing = [w for w in need if not (weights / w).exists()]
    print(f"  weights    : {'all present' if not missing else 'MISSING ' + ', '.join(missing)}")
    if missing:
        print(f"               place them in {weights}")
        ok = False

    videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    print(f"  videos     : {len(videos)} in data/videos/"
          + (f"  ({', '.join(v.name for v in videos)})" if videos else ""))
    if not videos:
        print("               add at least one .mp4 - one camera is created per file")

    try:
        from urbantrace.db.client import ping
        info = ping()
        print(f"  database   : connected  ({info['public_tables']} tables)")
    except Exception as e:
        print(f"  database   : NOT connected ({type(e).__name__})")
        print("               set SUPABASE_* in .env; the dashboard needs it")
        ok = False

    try:
        from urbantrace.routing import describe as rdescribe
        r = rdescribe()
        print(f"  routing    : {'GraphHopper key set' if r['api_key_configured'] else 'no key - straight-line estimates'}"
              f"  ({r['cached_routes']} cached)")
    except Exception as e:
        print(f"  routing    : unavailable ({type(e).__name__})")

    dist = ROOT / "web-dist" / "index.html"
    print(f"  frontend   : {'built' if dist.exists() else 'NOT built - run: python run.py build-web'}")

    return ok


# ----------------------------------------------------------------- steps

def build_web() -> int:
    head("BUILD FRONTEND")
    web = ROOT / "web"
    if not web.exists():
        print("  web/ not found - skipping (the legacy dashboard will be served)")
        return 0
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        print("  npm not found. Install Node.js 18+ from https://nodejs.org")
        print("  The legacy dashboard at /legacy still works without it.")
        return 1
    if not (web / "node_modules").exists():
        if run([npm, "install"], cwd=web) != 0:
            return 1
    return run([npm, "run", "build"], cwd=web)


def process(save: bool = True, no_video: bool = False) -> int:
    head("PROCESS VIDEOS  (detection -> tracking -> OCR -> routes -> alerts)")
    if not sorted((ROOT / "data" / "videos").glob("*.*")):
        print("  No videos in data/videos/ - nothing to process. Skipping.")
        return 0
    cmd = [PY, str(ROOT / "scripts" / "run_all_cameras.py")]
    if save:
        cmd.append("--save")
    if no_video:
        cmd.append("--no-video")
    return run(cmd)


def serve(host: str, port: int) -> int:
    head(f"DASHBOARD  ->  http://{host}:{port}")
    print("  Live Monitor tab: press Start for real-time detection.")
    print("  Ctrl+C to stop.\n")
    return run([PY, str(ROOT / "scripts" / "serve.py"),
                "--host", host, "--port", str(port)])


def selftest() -> int:
    head("SELF-TEST")
    rc = run([PY, "-m", "pytest", "tests/", "-q"])
    rc |= run([PY, str(ROOT / "scripts" / "verify_system.py")])
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="all",
                    choices=["all", "serve", "process", "check", "build-web", "test"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-save", action="store_true",
                    help="process without writing the run to the database")
    ap.add_argument("--no-video", action="store_true",
                    help="skip writing annotated output videos (faster)")
    a = ap.parse_args()

    if a.step == "check":
        return 0 if check() else 1
    if a.step == "build-web":
        return build_web()
    if a.step == "test":
        return selftest()
    if a.step == "process":
        return process(save=not a.no_save, no_video=a.no_video)
    if a.step == "serve":
        return serve(a.host, a.port)

    # all
    if not check():
        print("\nFix the items marked MISSING above, then run again.")
        return 1
    if not (ROOT / "web-dist" / "index.html").exists():
        build_web()
    rc = process(save=not a.no_save, no_video=a.no_video)
    if rc != 0:
        print("\nProcessing failed. The dashboard would have nothing to show.")
        return rc
    return serve(a.host, a.port)


if __name__ == "__main__":
    raise SystemExit(main())
