"""Worker membership and camera assignment.

Ingest runs as several worker processes against one database. Each worker
registers itself and heartbeats into `net_workers`; every worker reads the same
list of live workers and works out, independently, which cameras are its own.
There is no coordinator to fail.

Assignment is rendezvous (highest-random-weight) hashing with a load bound:

* every worker computes the same answer from the same inputs;
* groups prefer their highest-ranked worker, so a membership change mostly moves
  the groups that worker won or lost, not a full reshuffle;
* no worker is given more than 1.25x the average, so load stays even even when
  the cameras are clumped around a few busy junctions.

The hashed key is the camera's *affinity*, not its id: its sector, else a grid
cell of `shard_cell_km`. Cameras that are neighbours therefore land on the same
worker, which is what lets that worker's identity index see a vehicle at both
and measure the leg between them. Cameras on either side of a cell edge can be
owned by different workers; a leg between them is not recorded in `net_legs`,
though trajectory queries, which join sightings in SQL, still show it.

During a rebalance a camera can be read by two workers for up to one heartbeat
interval. That can duplicate a sighting for a few seconds and does nothing else.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import socket
import threading
import time
import uuid
from typing import Iterable, Protocol

from .config import SCALE

logger = logging.getLogger("urbantrace.scale.cluster")


def default_worker_id() -> str:
    return SCALE.worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def affinity_key(sector: str, latitude: float, longitude: float,
                 cell_km: float | None = None) -> str:
    """The unit that is assigned to a worker as a whole."""
    if sector:
        return f"sector:{sector}"
    cell = max(0.5, cell_km or SCALE.shard_cell_km)
    dlat = cell / 110.574
    dlon = cell / (111.320 * max(0.01, math.cos(math.radians(latitude))))
    return f"cell:{math.floor(latitude / dlat)}:{math.floor(longitude / dlon)}"


def _weight(worker_id: str, key: str) -> int:
    d = hashlib.blake2b(f"{worker_id}\x00{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big")


def owner_of(key: str, workers: Iterable[str]) -> str | None:
    """The worker that wins `key`. Ties break on worker id for determinism."""
    best, best_w = None, -1
    for w in workers:
        wt = _weight(w, key)
        if wt > best_w or (wt == best_w and best is not None and w < best):
            best, best_w = w, wt
    return best


def assign(cameras: Iterable[tuple[str, str]], workers: list[str], me: str,
           slack: float = 1.25) -> set[str]:
    """Camera ids owned by `me`. `cameras` yields (camera_id, affinity_key).

    Pure rendezvous hashing balances well only when there are many units. A
    city has few, uneven ones (a busy junction can hold a dozen cameras), and
    plain hashing can leave a worker with nothing. So affinity groups are placed
    largest first, each on its highest-ranked worker that is still under
    `slack` times the average load, falling back to the least loaded.

    Every worker runs the same deterministic procedure on the same inputs, so
    they agree without talking to each other.
    """
    if me not in workers:
        return set()
    groups: dict[str, list[str]] = {}
    for cid, key in cameras:
        groups.setdefault(key, []).append(cid)
    total = sum(len(v) for v in groups.values())
    cap = max(1, math.ceil(slack * total / len(workers)))
    load = {w: 0 for w in workers}
    mine: set[str] = set()
    for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ranked = sorted(workers, key=lambda w: (-_weight(w, key), w))
        pick = next((w for w in ranked if load[w] + len(members) <= cap), None)
        if pick is None:
            pick = min(workers, key=lambda w: (load[w], w))
        load[pick] += len(members)
        if pick == me:
            mine.update(members)
    return mine


class MembershipStore(Protocol):
    def upsert(self, worker_id: str, host: str, stats: dict) -> None: ...
    def live(self, ttl_s: float) -> list[dict]: ...
    def remove(self, worker_id: str) -> None: ...


class PgMembershipStore:
    """`net_workers` in Postgres. Time comes from the database, not the host,
    so clock drift between workers cannot make a healthy one look dead."""

    def upsert(self, worker_id: str, host: str, stats: dict) -> None:
        import json

        from ..db.client import cursor
        with cursor() as cur:
            cur.execute("""
                insert into net_workers (worker_id, host, started_at, heartbeat_at, stats)
                values (%s, %s, now(), now(), %s::jsonb)
                on conflict (worker_id) do update
                  set heartbeat_at = now(), host = excluded.host, stats = excluded.stats
            """, (worker_id, host, json.dumps(stats)))

    def live(self, ttl_s: float) -> list[dict]:
        from ..db import repository as repo
        return repo._fetch("""
            select worker_id, host, started_at, heartbeat_at, stats,
                   extract(epoch from (now() - heartbeat_at)) as age_s
            from net_workers
            where heartbeat_at > now() - make_interval(secs => %s)
            order by worker_id""", (ttl_s,))

    def remove(self, worker_id: str) -> None:
        from ..db.client import cursor
        with cursor() as cur:
            cur.execute("delete from net_workers where worker_id = %s", (worker_id,))


class Membership:
    """This worker's view of the cluster."""

    def __init__(self, store: MembershipStore | None = None, worker_id: str | None = None):
        self.store = store or PgMembershipStore()
        self.worker_id = worker_id or default_worker_id()
        self.host = socket.gethostname()
        self._workers: list[str] = [self.worker_id]
        self._lock = threading.Lock()
        self.last_ok_at = 0.0
        self.failures = 0
        self.last_error: str | None = None

    def beat(self, stats: dict) -> list[str]:
        """Heartbeat, then return the live worker ids.

        If the database is unreachable the previous membership is kept, so a
        brief outage does not make every worker abandon its cameras.
        """
        try:
            self.store.upsert(self.worker_id, self.host, stats)
            rows = self.store.live(SCALE.worker_ttl_s)
            ids = sorted({r["worker_id"] for r in rows} | {self.worker_id})
            with self._lock:
                self._workers = ids
            self.last_ok_at = time.time()
            return ids
        except Exception as e:                          # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("membership heartbeat failed: %s", self.last_error)
            with self._lock:
                return list(self._workers)

    @property
    def workers(self) -> list[str]:
        with self._lock:
            return list(self._workers)

    def leave(self) -> None:
        try:
            self.store.remove(self.worker_id)
        except Exception as e:                          # noqa: BLE001
            logger.warning("could not deregister worker: %s", e)
