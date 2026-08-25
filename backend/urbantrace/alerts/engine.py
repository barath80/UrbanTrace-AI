"""Watchlist alerting.

The alerts table and its API endpoint existed, but nothing generated rows: the
only alerts ever written were a synthetic one from the verification script. This
module is the engine behind them.

TWO THINGS IT GETS RIGHT THAT A NAIVE STRING MATCH DOES NOT

  1. It matches on the VEHICLE, not the OCR text. Association merges variants -
     TN01AB1234 read as TN01A8123 at the second camera is one car - so matching
     watchlist entries against raw observation text misses exactly the sightings
     that matter. Alerts fire per sighting, but membership is decided by the
     resolved plate.

  2. It never invents a watchlist. Alerts fire only for plates the operator put
     on the list. An empty list means no alerts, which is the correct and safe
     default - a system that auto-seeds its own watchlist from what it happens
     to have seen is not an alerting system.

Severity reflects the confidence of the read, because a low-confidence hit on a
blacklisted plate is a lead to check, not a fact to act on.
"""
from __future__ import annotations

from ..core.cameras import CAMERAS

# An OCR read at or above this confidence is treated as a firm identification.
FIRM_CONFIDENCE = 0.85
# Below this the hit is reported but flagged as needing human confirmation.
WEAK_CONFIDENCE = 0.60


def load_watchlist_file(path) -> set:
    """Plates from a watchlist text file: one per line, `#` comments ignored."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper().replace(" ", "")
        if line:
            out.add(line)
    return out


def _severity(conf: float) -> tuple:
    if conf >= FIRM_CONFIDENCE:
        return "HIGH", "firm identification"
    if conf >= WEAK_CONFIDENCE:
        return "MEDIUM", "moderate-confidence read - confirm against the evidence crop"
    return "LOW", "low-confidence read - likely needs human confirmation"


def watchlist_alerts(vehicles, watchlist) -> list:
    """One alert per sighting of a watchlisted vehicle.

    `vehicles` are GlobalVehicle objects or dicts (they carry the resolved plate
    and every observation behind it). `watchlist` is a set of plate strings.
    Returns dicts shaped for repository.save_alerts.
    """
    wl = {p.upper().replace(" ", "") for p in watchlist if p}
    if not wl:
        return []

    out = []
    for v in vehicles:
        v = v if isinstance(v, dict) else v.to_dict()
        plate = (v.get("plate_number") or "").upper()
        # A variant that was merged into this vehicle also counts as a match:
        # the operator listed the car, not one particular OCR spelling.
        variants = {plate, *(s.upper() for s in v.get("merged_variants") or [])}
        hit = variants & wl
        if not hit:
            continue
        matched_on = sorted(hit)[0]

        for o in v.get("observations", []):
            conf = float(o.get("ocr_confidence") or 0.0)
            sev, note = _severity(conf)
            cam = o.get("camera_id")
            location = CAMERAS[cam].name if cam in CAMERAS else cam
            via = "" if matched_on == plate else \
                f" (matched via merged OCR variant {matched_on})"
            read_as = (o.get("plate_number") or plate).upper()
            read_note = "" if read_as == plate else f", read here as {read_as}"
            out.append({
                "alert_type": "WATCHLIST_MATCH",
                "severity": sev,
                "plate": plate,
                "camera": cam,
                "location": location,
                "timestamp": o.get("timestamp"),
                "ocr_confidence": round(conf, 4),
                "evidence": o.get("evidence_crop"),
                "reason": (f"Watchlisted plate {plate} seen at {cam} "
                           f"({location}){via}{read_note}. "
                           f"OCR confidence {conf:.3f} - {note}."),
            })

    out.sort(key=lambda a: (a.get("timestamp") or "", a.get("camera") or ""))
    return out


def summarize(alerts: list) -> dict:
    by_sev, by_plate = {}, {}
    for a in alerts:
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
        by_plate[a["plate"]] = by_plate.get(a["plate"], 0) + 1
    return {"total": len(alerts), "by_severity": by_sev,
            "plates_hit": sorted(by_plate), "sightings_per_plate": by_plate}
