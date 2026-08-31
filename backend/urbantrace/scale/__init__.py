"""City-scale ingest, identity resolution, analytics and alerting.

The batch pipeline in `urbantrace.vision` processes a folder of videos into a
run. This package runs a live camera network instead: thousands of cameras,
one shared set of models, continuous sightings, and analytics served from
Postgres rather than from process memory.
"""
from .config import SCALE                                    # noqa: F401
from .registry import REGISTRY, CameraNode, CameraRegistry   # noqa: F401
from .identity import IdentityIndex, Sighting, same_vehicle  # noqa: F401
from .service import SERVICE, IngestService, shard_of        # noqa: F401
