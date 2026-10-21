"""Supabase / PostgreSQL connection.

Credentials come from `.env` (git-ignored) or the environment - never from
source. Two gotchas this module handles for you:

1. The Supabase DIRECT host (`db.<ref>.supabase.co`) resolves to IPv6 only.
   Networks without IPv6 egress cannot reach it, so the IPv4 **pooler** is the
   default and the direct host is only a fallback.
2. Passwords containing `@`, `:` or `/` break URI parsing. The DSN is assembled
   with proper percent-encoding rather than string-concatenated.
"""
from __future__ import annotations

import os
import threading
import urllib.parse as up
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"


def _load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items()
                if k.startswith(("SUPABASE_", "DATABASE_URL", "UT_DB_"))})
    return env


def database_url() -> str | None:
    """A plain Postgres URL, for any self-hosted or managed PostGIS server.

    When DATABASE_URL is set it wins over the Supabase settings, so the same
    code runs against the compose-managed PostGIS container in production and
    Supabase in development.
    """
    return _load_env().get("DATABASE_URL") or None


def _dsn_candidates() -> list[tuple[str, str]]:
    """(dsn, sslmode) pairs to try, most preferred first."""
    url = database_url()
    if url:
        return [(url, _load_env().get("UT_DB_SSLMODE", "prefer"))]
    cfg = get_config()
    order = [cfg.dsn()] + [d for d in (cfg.pooler_dsn, cfg.direct_dsn) if d != cfg.dsn()]
    return [(d, "require") for d in order]


@dataclass(frozen=True)
class DbConfig:
    project_ref: str
    password: str
    region: str = "ap-south-1"
    pooler_port: int = 5432        # 5432 = session pooler, 6543 = transaction pooler
    prefer: str = "pooler"         # "pooler" | "direct"

    @property
    def pooler_dsn(self) -> str:
        user = f"postgres.{self.project_ref}"
        host = f"aws-0-{self.region}.pooler.supabase.com"
        return (f"postgresql://{up.quote(user)}:{up.quote_plus(self.password)}"
                f"@{host}:{self.pooler_port}/postgres")

    @property
    def direct_dsn(self) -> str:
        host = f"db.{self.project_ref}.supabase.co"
        return (f"postgresql://postgres:{up.quote_plus(self.password)}"
                f"@{host}:5432/postgres")

    def dsn(self) -> str:
        return self.pooler_dsn if self.prefer == "pooler" else self.direct_dsn

    def redacted(self) -> str:
        return self.dsn().replace(up.quote_plus(self.password), "***")


@lru_cache(maxsize=1)
def get_config() -> DbConfig:
    env = _load_env()
    url = env.get("SUPABASE_URL", "")
    ref = env.get("SUPABASE_PROJECT_REF") or (
        up.urlparse(url).hostname.split(".")[0] if url else "")
    pwd = env.get("SUPABASE_DB_PASSWORD", "")
    if not ref or not pwd:
        raise RuntimeError(
            "Supabase not configured. Add SUPABASE_URL and SUPABASE_DB_PASSWORD "
            f"to {ENV_FILE} (see .env.example).")
    return DbConfig(
        project_ref=ref, password=pwd,
        region=env.get("SUPABASE_REGION", "ap-south-1"),
        pooler_port=int(env.get("SUPABASE_POOLER_PORT", "5432")),
        prefer=env.get("SUPABASE_PREFER", "pooler"),
    )


def connect(timeout: int = 20):
    """Open a psycopg2 connection to the configured database."""
    import psycopg2
    last = None
    for dsn, sslmode in _dsn_candidates():
        try:
            return psycopg2.connect(dsn, connect_timeout=timeout, sslmode=sslmode)
        except Exception as e:  # try the next transport
            last = e
    raise ConnectionError(
        f"Could not reach the database. Last error: {type(last).__name__}: {last}\n"
        "On Supabase, an IPv6 error means the direct host is IPv6-only - use the pooler."
    ) from last


# ---------------------------------------------------------- pooling
# Supabase lives in ap-south-1 and every connection costs a TCP + TLS
# handshake. Opening one per query made a single dashboard request that runs
# seven queries take six seconds, nearly all of it handshaking. Connections are
# therefore pooled per process and reused.

_POOL = None
_POOL_LOCK = threading.Lock()
POOL_MIN = 1
POOL_MAX = int(os.environ.get("UT_DB_POOL_MAX", "8"))
# psycopg2's pool RAISES when it is empty instead of waiting. FastAPI runs sync
# handlers on a 40-thread pool, so a burst larger than POOL_MAX turned into 500s
# (measured: 30 concurrent dashboard requests against 8 connections). The
# semaphore makes the extra callers queue for a connection instead.
POOL_WAIT_S = float(os.environ.get("UT_DB_POOL_WAIT_S", "30"))
_SLOTS = threading.BoundedSemaphore(POOL_MAX)


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from psycopg2.pool import ThreadedConnectionPool
                last = None
                for dsn, sslmode in _dsn_candidates():
                    try:
                        _POOL = ThreadedConnectionPool(
                            POOL_MIN, POOL_MAX, dsn,
                            connect_timeout=20, sslmode=sslmode)
                        break
                    except Exception as e:
                        last = e
                if _POOL is None:
                    raise ConnectionError(
                        f"Could not reach the database. Last error: "
                        f"{type(last).__name__}: {last}") from last
    return _POOL


@contextmanager
def pooled(commit: bool = True):
    """A pooled connection. Broken connections are discarded, not returned."""
    pool = _pool()
    if not _SLOTS.acquire(timeout=POOL_WAIT_S):
        raise TimeoutError(f"no database connection free after {POOL_WAIT_S:g}s "
                           f"(UT_DB_POOL_MAX={POOL_MAX})")
    try:
        conn = pool.getconn()
        bad = False
        try:
            yield conn
            if commit:
                conn.commit()
        except Exception:
            bad = True
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                pool.putconn(conn, close=bad)
            except Exception:
                pass
    finally:
        _SLOTS.release()


def close_pool() -> None:
    """Drop every pooled connection. Used by tests and on shutdown."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            try:
                _POOL.closeall()
            finally:
                _POOL = None


@contextmanager
def cursor(commit: bool = True):
    with pooled(commit=commit) as conn:
        with conn.cursor() as cur:
            yield cur


def _redacted_target() -> str:
    url = database_url()
    if url:
        p = up.urlparse(url)
        return f"{p.scheme}://{p.hostname}:{p.port or 5432}{p.path}"
    return get_config().redacted()


def ping() -> dict:
    """Connectivity + server info, for the notebook's status cell."""
    with cursor(commit=False) as cur:
        cur.execute("select version(), current_database(), current_user, now()")
        v, db, user, now = cur.fetchone()
        cur.execute("select count(*) from information_schema.tables "
                    "where table_schema='public'")
        n_tables = cur.fetchone()[0]
    return {"connected": True, "dsn": _redacted_target(), "server": v.split(",")[0],
            "database": db, "user": user, "server_time": str(now),
            "public_tables": n_tables}


def apply_schema(verbose: bool = True) -> int:
    """Create every table/view. Idempotent - safe to re-run."""
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with cursor() as cur:
        cur.execute(sql)
        cur.execute("select table_name from information_schema.tables "
                    "where table_schema='public' order by 1")
        tables = [r[0] for r in cur.fetchall()]
    if verbose:
        print(f"schema applied - {len(tables)} tables/views in public:")
        for t in tables:
            print("   ", t)
    return len(tables)


def reset_schema(confirm: bool = False) -> None:
    """DESTRUCTIVE: drop every UrbanTrace table. Requires confirm=True."""
    if not confirm:
        raise RuntimeError("reset_schema(confirm=True) required - this deletes all data")
    objs = ["v_plate_timeline", "v_camera_flows", "v_camera_counts"]
    tables = ["route_legs", "vehicle_observations", "global_vehicles",
              "plate_readings", "observations", "camera_stats", "videos",
              "camera_edges", "cameras", "alerts", "watchlist", "runs"]
    with cursor() as cur:
        for v in objs:
            cur.execute(f"drop view if exists {v} cascade")
        for t in tables:
            cur.execute(f"drop table if exists {t} cascade")
    print("schema dropped")
