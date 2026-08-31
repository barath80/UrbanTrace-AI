"""Measure what this machine can actually carry.

Answers the only question that matters when sizing a city deployment: how many
cameras can one GPU serve, and where does the platform stop being the
bottleneck?

    python scripts/benchmark_scale.py                  # everything
    python scripts/benchmark_scale.py --skip-gpu       # CPU-only parts
    python scripts/benchmark_scale.py --cameras 2000

Nothing here is simulated. The registry and identity benchmarks run the real
data structures, and the throughput benchmark runs the real detectors over the
real videos in data/videos/.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

LETTERS, DIGITS = "ABCDEFGHJKMNPRSTUVWXYZ", "0123456789"
CONFUSIONS = {"0": "OQD", "O": "0", "1": "IL", "I": "1", "8": "B",
              "B": "8", "5": "S", "S": "5", "2": "Z", "Z": "2"}


def _plate(rng: random.Random) -> str:
    return ("".join(rng.choice(LETTERS) for _ in range(2))
            + "".join(rng.choice(DIGITS) for _ in range(2))
            + "".join(rng.choice(LETTERS) for _ in range(2))
            + "".join(rng.choice(DIGITS) for _ in range(4)))


def _misread(plate: str, rng: random.Random, rate: float = 0.45) -> str:
    """Apply a realistic OCR confusion to some fraction of readings."""
    if rng.random() > rate:
        return plate
    positions = [i for i, c in enumerate(plate) if c in CONFUSIONS]
    if not positions:
        return plate
    i = rng.choice(positions)
    return plate[:i] + rng.choice(CONFUSIONS[plate[i]]) + plate[i + 1:]


def bench_registry(sizes) -> None:
    from urbantrace.scale.registry import CameraRegistry

    print("\n== camera registry ==")
    print(f"  {'cameras':>8}  {'build':>9}  {'edges':>9}  {'all-pairs':>13}  {'near()':>9}")
    rng = random.Random(7)
    for n in sizes:
        reg = CameraRegistry(cell_km=2.0)
        reg.load([{"camera_id": f"CAM_{i:06d}", "name": f"N{i}",
                   "latitude": 12.80 + rng.random() * 0.55,
                   "longitude": 79.95 + rng.random() * 0.55} for i in range(n)])
        t = time.perf_counter()
        info = reg.build_graph()
        build = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(1000):
            reg.near(12.80 + rng.random() * 0.55, 79.95 + rng.random() * 0.55, 5.0, limit=8)
        near_us = (time.perf_counter() - t) / 1000 * 1e6

        print(f"  {n:8,}  {build*1000:7.0f}ms  {info['edges']:9,}  "
              f"{n*(n-1)//2:13,}  {near_us:7.0f}us")


def bench_identity(vehicle_counts, cameras: int, per_vehicle: int) -> None:
    from urbantrace.scale.identity import IdentityIndex, Sighting

    print(f"\n== identity resolution ({cameras} cameras, "
          f"{per_vehicle} sightings/vehicle, 45% OCR noise) ==")
    print(f"  {'vehicles':>8}  {'sightings':>10}  {'ingest':>8}  {'rate':>12}  "
          f"{'clusters':>9}  {'cmp/insert':>10}")
    base = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for n in vehicle_counts:
        rng = random.Random(11)
        plates = [_plate(rng) for _ in range(n)]
        events = []
        for vi, p in enumerate(plates):
            for k in range(per_vehicle):
                events.append((_misread(p, rng), f"CAM_{rng.randrange(cameras):05d}",
                               vi * 0.01 + k * 7.0))
        rng.shuffle(events)

        idx = IdentityIndex()
        t = time.perf_counter()
        for i, (p, cam, mins) in enumerate(events):
            idx.add(Sighting(plate=p, camera_id=cam,
                             observed_at=base + timedelta(minutes=mins),
                             confidence=0.85, track_id=i,
                             latitude=13.0, longitude=80.0))
        el = time.perf_counter() - t
        st = idx.stats()
        print(f"  {n:8,}  {len(events):10,}  {el:6.2f}s  {len(events)/el:9,.0f}/s  "
              f"{st['clusters']:9,}  {st['comparisons_per_add']:10.2f}")
        # Cluster count should track the true vehicle count; a large deviation
        # means the resolver is splitting or merging identities.
        drift = abs(st["clusters"] - n) / n
        if drift > 0.02:
            print(f"      WARNING: cluster count is {drift*100:.1f}% off the true "
                  f"vehicle count ({st['clusters']:,} vs {n:,})")


def bench_throughput(seconds: float, cameras: int) -> None:
    import cv2

    from urbantrace.scale.config import SCALE
    from urbantrace.scale.identity import IdentityIndex
    from urbantrace.scale.inference import FrameBus, FrameItem, InferencePool, SharedModels
    from urbantrace.scale.registry import REGISTRY

    videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    if not videos:
        print("\n== throughput ==\n  no videos in data/videos/; skipped")
        return

    print(f"\n== end-to-end throughput ({cameras} cameras, real footage) ==")
    REGISTRY.clear()
    REGISTRY.load([{"camera_id": f"CAM_{i:05d}", "name": f"N{i}",
                    "latitude": 13.0 + (i % 40) * 0.004,
                    "longitude": 80.1 + (i // 40) * 0.004} for i in range(cameras)])
    REGISTRY.build_graph()

    clips = []
    for v in videos[:4]:
        cap = cv2.VideoCapture(str(v))
        frames = []
        while len(frames) < 150:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if frames:
            clips.append(frames)
    if not clips:
        print("  videos could not be decoded; skipped")
        return

    idx = IdentityIndex()
    bus = FrameBus(target_fps=0)          # measure raw capacity, not the cap
    pool = InferencePool(models=SharedModels(), bus=bus, on_sighting=idx.add)
    pool.start()

    t0 = time.time()
    i = 0
    while time.time() - t0 < seconds:
        for c in range(cameras):
            seq = clips[c % len(clips)]
            bus.submit(FrameItem(f"CAM_{c:05d}", seq[i % len(seq)],
                                 datetime.now(timezone.utc), (i % len(seq)) / 24.0, i))
        i += 1
        while bus.stats()["frames_pending"] > cameras:
            time.sleep(0.01)
    while bus.stats()["frames_pending"] > 0:
        time.sleep(0.2)
    time.sleep(1.0)
    el = time.time() - t0
    st = pool.stats()
    pool.stop()

    m = st["models"]
    fps = m["frames_processed"] / el
    print(f"  frames analysed      : {m['frames_processed']:,} in {el:.0f}s")
    print(f"  throughput           : {fps:,.1f} analysed frames/sec")
    print(f"  mean detector batch  : {m['mean_batch']}")
    print(f"  plate passes skipped : {m['plate_frames_skipped']:,} "
          f"({100*m['plate_frames_skipped']/max(1, m['frames_processed']):.0f}%)")
    print(f"  OCR crops            : {st['ocr_calls']:,}")
    print(f"  sightings emitted    : {st['sightings_emitted']:,}")
    print(f"  identity clusters    : {idx.stats()['clusters']:,}")
    print("\n  Cameras one GPU sustains, by analysed frames per camera per second:")
    for target in (1.0, 2.0, 4.0, 6.0):
        print(f"    {target:>4.1f} fps/camera -> {fps/target:7,.0f} cameras")
    print(f"\n  Configured target is {SCALE.target_fps_per_camera} fps/camera "
          f"-> {fps/max(0.1, SCALE.target_fps_per_camera):,.0f} cameras on this GPU.")
    print("  Scale beyond that horizontally: run N sharded ingest processes "
          "(scale.service.shard_of)")
    print("  against the same database; each owns a disjoint slice of the estate.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", type=int, default=40,
                    help="cameras for the throughput benchmark")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--skip-gpu", action="store_true",
                    help="skip the detector benchmark")
    ap.add_argument("--registry-sizes", type=int, nargs="*",
                    default=[1000, 5000, 10000])
    ap.add_argument("--identity-sizes", type=int, nargs="*",
                    default=[2000, 10000, 40000])
    args = ap.parse_args()

    print("UrbanTrace AI -- scale benchmark")
    try:
        from urbantrace.core.runtime import describe
        rt = describe()
        print(f"  device: {rt.get('device')}  {rt.get('gpu_name') or ''}")
    except Exception:
        pass

    bench_registry(args.registry_sizes)
    bench_identity(args.identity_sizes, cameras=1000, per_vehicle=6)
    if not args.skip_gpu:
        bench_throughput(args.seconds, args.cameras)


if __name__ == "__main__":
    main()
