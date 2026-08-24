"""Central configuration for UrbanTrace AI.

All tunables live here so that deployment scale (5 cameras -> thousands) is a
configuration concern, not a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_f(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_i(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class OCRSettings:
    """OCR / ANPR engine configuration.

    ``backend`` is resolved at runtime; if the requested engine is not
    installed the pipeline reports it as UNAVAILABLE rather than silently
    substituting fabricated results.
    """

    backend: str = field(default_factory=lambda: _env("UT_OCR_BACKEND", "auto"))
    min_confidence: float = field(default_factory=lambda: _env_f("UT_OCR_MIN_CONF", 0.45))
    # Below this an observation is stored but marked low-confidence and is not
    # allowed to seed a new vehicle identity on its own.
    identity_confidence: float = field(default_factory=lambda: _env_f("UT_OCR_IDENTITY_CONF", 0.70))
    plate_regex_region: str = field(default_factory=lambda: _env("UT_PLATE_REGION", "IN"))


@dataclass(frozen=True)
class AssociationSettings:
    """Cross-camera association and spatio-temporal validation thresholds."""

    # Maximum plausible road speed used for the hard feasibility gate (km/h).
    max_plausible_speed_kmh: float = field(default_factory=lambda: _env_f("UT_MAX_SPEED", 140.0))
    # Speed above which a transition is suspicious but not impossible.
    warn_speed_kmh: float = field(default_factory=lambda: _env_f("UT_WARN_SPEED", 100.0))
    # Minimum realistic speed; slower implies the vehicle stopped in between,
    # which is allowed but breaks the trajectory into legs beyond this gap.
    trajectory_gap_minutes: float = field(default_factory=lambda: _env_f("UT_TRAJ_GAP_MIN", 90.0))
    # Minimum plate similarity for two observations to be candidates.
    min_plate_similarity: float = field(default_factory=lambda: _env_f("UT_MIN_PLATE_SIM", 0.82))
    # Weights of the association score components. Must sum to 1.0.
    weights: dict = field(default_factory=lambda: {
        "plate_similarity": 0.40,
        "ocr_confidence": 0.15,
        "travel_time_plausibility": 0.25,
        "graph_connectivity": 0.12,
        "direction_consistency": 0.08,
    })


@dataclass(frozen=True)
class AnalyticsSettings:
    window_minutes: int = field(default_factory=lambda: _env_i("UT_WINDOW_MIN", 15))
    baseline_days: int = field(default_factory=lambda: _env_i("UT_BASELINE_DAYS", 28))
    # Density classification thresholds in vehicles/minute per camera.
    density_bands: tuple = (10.0, 30.0, 60.0)  # low|moderate|heavy|severe
    # Congestion is flagged when observed speed drops below this fraction of
    # the historical baseline for the same road segment and time-of-day slot.
    congestion_speed_ratio: float = field(default_factory=lambda: _env_f("UT_CONGESTION_RATIO", 0.60))
    congestion_traveltime_ratio: float = field(default_factory=lambda: _env_f("UT_CONGESTION_TT_RATIO", 1.40))
    min_samples_for_baseline: int = field(default_factory=lambda: _env_i("UT_MIN_BASELINE_SAMPLES", 5))


@dataclass(frozen=True)
class SecuritySettings:
    secret_key: str = field(default_factory=lambda: _env("UT_SECRET_KEY", "dev-only-insecure-change-me"))
    algorithm: str = "HS256"
    token_ttl_minutes: int = field(default_factory=lambda: _env_i("UT_TOKEN_TTL_MIN", 480))
    # Retention for raw evidence images, in days. 0 disables automatic purge.
    evidence_retention_days: int = field(default_factory=lambda: _env_i("UT_EVIDENCE_RETENTION_DAYS", 90))
    observation_retention_days: int = field(default_factory=lambda: _env_i("UT_OBS_RETENTION_DAYS", 365))


@dataclass(frozen=True)
class Settings:
    app_name: str = "UrbanTrace AI"
    tagline: str = "From License Plate Recognition to City-Wide Mobility Intelligence"
    version: str = "0.1.0-prototype"
    database_url: str = field(default_factory=lambda: _env("UT_DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'urbantrace.db'}"))
    evidence_dir: Path = field(default_factory=lambda: Path(_env("UT_EVIDENCE_DIR", str(ROOT / "data" / "evidence"))))
    frontend_dir: Path = field(default_factory=lambda: ROOT / "frontend")
    # A camera is considered stale (NO_RECENT_DETECTION) after this long.
    camera_stale_minutes: int = field(default_factory=lambda: _env_i("UT_CAMERA_STALE_MIN", 10))
    camera_offline_minutes: int = field(default_factory=lambda: _env_i("UT_CAMERA_OFFLINE_MIN", 30))
    ocr: OCRSettings = field(default_factory=OCRSettings)
    association: AssociationSettings = field(default_factory=AssociationSettings)
    analytics: AnalyticsSettings = field(default_factory=AnalyticsSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)


settings = Settings()
settings.evidence_dir.mkdir(parents=True, exist_ok=True)
