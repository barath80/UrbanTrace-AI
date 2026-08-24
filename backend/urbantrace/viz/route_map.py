"""Route visualisation: schematic corridor sketch + real GIS map.

Two complementary views of the same reconstructed routes:
  * `sketch_route_map`  - matplotlib schematic, renders inline in Jupyter.
  * `gis_route_map`     - folium/Leaflet map on real Chennai geography.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..core.cameras import CAMERAS, get_edge

STATUS_COLORS = {
    "COMPLETE_CORRIDOR": "#1a9850",   # green - traversed the whole corridor
    "PARTIAL_ROUTE": "#fdae61",       # amber - stopped partway
    "DROPPED_AT_ORIGIN": "#d73027",   # red   - never left the first camera
    "SINGLE_CAMERA_NETWORK": "#9aa5b1",  # grey - only one camera, no route possible
}


def _as_dicts(vehicles):
    return [v if isinstance(v, dict) else v.to_dict() for v in vehicles]


def flow_counts(vehicles) -> Counter:
    """Camera-to-camera vehicle flow counts (origin-destination edges)."""
    flows = Counter()
    for v in _as_dicts(vehicles):
        route = v.get("route", [])
        for a, b in zip(route, route[1:]):
            flows[(a, b)] += 1
    return flows


def sketch_route_map(vehicles, out_path=None, figsize=None, max_labels=16):
    """Schematic corridor map for ANY number of cameras."""
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vd = _as_dicts(vehicles)
    order = list(CAMERAS)
    n = len(order)
    if n == 0:
        raise ValueError("No cameras configured - add videos to data/videos/")

    # Node spacing shrinks as cameras are added so the row always fits.
    span = max(8.0, min(4.0 * n, 22.0))
    step = span / max(1, n - 1) if n > 1 else 0.0
    pos = {cam: (i * step, 0.0) for i, cam in enumerate(order)}

    shown = vd[:max_labels]
    if figsize is None:
        figsize = (max(13, 2.2 * n + 6), 6 + 0.34 * max(1, len(shown)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0f1419"); fig.patch.set_facecolor("#0f1419")

    flows = flow_counts(vd)
    max_flow = max(flows.values()) if flows else 1
    gap = step * 0.13 if n > 1 else 0.4

    for (a, b), cnt in flows.items():
        if a not in pos or b not in pos:
            continue
        x1, _ = pos[a]; x2, _ = pos[b]
        # Skip-edges arc above so they do not overlap consecutive edges.
        ia, ib = order.index(a), order.index(b)
        rad = 0.0 if abs(ib - ia) == 1 else 0.25 * (ib - ia)
        ax.annotate("", xy=(x2 - gap, 0), xytext=(x1 + gap, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#4da6ff", alpha=.85,
                                    lw=1.5 + 7.0 * (cnt / max_flow),
                                    connectionstyle=f"arc3,rad={rad}"))
        edge = get_edge(a, b)
        yl = 0.30 + (0.30 * abs(ib - ia) if abs(ib - ia) > 1 else 0.0)
        ax.text((x1 + x2) / 2, yl + 0.12, f"{cnt} vehicle(s)", ha="center",
                color="#8ecfff", fontsize=10, fontweight="bold")
        if edge:
            ax.text((x1 + x2) / 2, yl - 0.04, f"{edge.road_distance_km:.1f} km",
                    ha="center", color="#6b8299", fontsize=8)

    for cam, (x, _y) in pos.items():
        c = CAMERAS[cam]
        ax.scatter([x], [0], s=1200, c="#ffd166", edgecolors="#ffffff",
                   linewidths=2, zorder=5)
        ax.text(x, 0, cam.replace("CAM_", "C"), ha="center", va="center",
                fontsize=11, fontweight="bold", color="#0f1419", zorder=6)
        ax.text(x, -0.42, c.name, ha="center", color="#ffffff",
                fontsize=9, fontweight="bold", rotation=0 if n <= 6 else 18)
        ax.text(x, -0.62, f"{c.latitude:.4f}, {c.longitude:.4f}",
                ha="center", color="#6b8299", fontsize=7)

    # Per-vehicle rows.
    top = -1.15
    cols = (0.0, span * 0.26, span * 0.66, span * 0.87)
    for hx, htxt in zip(cols, ("PLATE", "ROUTE", "STATUS", "DURATION")):
        ax.text(hx, top + 0.30, htxt, color="#6b8299", fontsize=8, fontweight="bold")
    for i, v in enumerate(shown):
        y = top - i * 0.34
        colour = STATUS_COLORS.get(v.get("route_status", ""), "#999999")
        ax.text(cols[0], y, v.get("plate_number", "?"), color="#ffffff",
                fontsize=9, fontweight="bold", family="monospace")
        ax.text(cols[1], y, " -> ".join(v.get("route", [])), color=colour,
                fontsize=8.5, family="monospace")
        ax.text(cols[2], y, v.get("route_status", "").replace("_", " ").title(),
                color=colour, fontsize=8.5)
        if v.get("camera_count", 0) > 1:
            ax.text(cols[3], y, f"{v.get('total_duration_minutes', 0):.1f} min",
                    color="#8ecfff", fontsize=8.5)
    if len(vd) > len(shown):
        ax.text(cols[0], top - len(shown) * 0.34,
                f"... +{len(vd) - len(shown)} more vehicles",
                color="#6b8299", fontsize=8.5, style="italic")

    handles = [plt.Line2D([], [], marker="s", ls="", color=c,
                          label=s.replace("_", " ").title())
               for s, c in STATUS_COLORS.items()]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True,
                    facecolor="#1b2530", edgecolor="#33465c", fontsize=8.5)
    for t in leg.get_texts():
        t.set_color("#dce6f0")

    ax.set_title(f"UrbanTrace AI - Vehicle Routes Across {n} Camera(s)",
                 color="#ffffff", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlim(-span * 0.12, span * 1.12)
    ax.set_ylim(top - max(1, len(shown)) * 0.34 - 0.3, 1.35)
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def gis_route_map(vehicles, out_path=None, highlight_plate: str | None = None):
    """Real-geography Leaflet map of cameras and reconstructed vehicle routes."""
    import folium

    vd = _as_dicts(vehicles)
    lats = [c.latitude for c in CAMERAS.values()]
    lons = [c.longitude for c in CAMERAS.values()]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=13, tiles="CartoDB dark_matter")

    flows = flow_counts(vd)
    max_flow = max(flows.values()) if flows else 1

    # Aggregate flow lines.
    for (a, b), n in flows.items():
        if a not in CAMERAS or b not in CAMERAS:
            continue
        ca, cb = CAMERAS[a], CAMERAS[b]
        edge = get_edge(a, b)
        folium.PolyLine(
            [(ca.latitude, ca.longitude), (cb.latitude, cb.longitude)],
            color="#4da6ff", weight=2 + 8 * (n / max_flow), opacity=0.75,
            tooltip=(f"{a} -> {b}: {n} vehicle(s)"
                     + (f", {edge.road_distance_km:.2f} km" if edge else "")),
        ).add_to(m)

    # A single highlighted vehicle route, drawn on top.
    if highlight_plate:
        for v in vd:
            if v.get("plate_number") != highlight_plate:
                continue
            pts = [(o["latitude"], o["longitude"]) for o in v.get("observations", [])]
            if len(pts) > 1:
                folium.PolyLine(pts, color="#ff4d4d", weight=5, opacity=0.95,
                                tooltip=f"{highlight_plate}: {' -> '.join(v['route'])}"
                                ).add_to(m)
            for o in v.get("observations", []):
                folium.CircleMarker(
                    [o["latitude"], o["longitude"]], radius=7,
                    color="#ff4d4d", fill=True, fill_opacity=0.9,
                    popup=folium.Popup(
                        f"<b>{o['plate_number']}</b><br>{o['camera_id']}<br>"
                        f"{o['timestamp']}<br>conf {o['ocr_confidence']:.2f}<br>"
                        f"track #{o['local_track_id']}", max_width=260),
                ).add_to(m)

    # Camera markers.
    for cam_id, c in CAMERAS.items():
        seen = sum(1 for v in vd for o in v.get("observations", [])
                   if o["camera_id"] == cam_id)
        folium.Marker(
            [c.latitude, c.longitude],
            tooltip=f"{cam_id} - {c.name}",
            popup=folium.Popup(
                f"<b>{cam_id}</b><br>{c.name}<br>{c.road}<br>"
                f"{c.latitude:.4f}, {c.longitude:.4f}<br>"
                f"observations: {seen}", max_width=280),
            icon=folium.Icon(color="orange", icon="video-camera", prefix="fa"),
        ).add_to(m)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m


def traffic_heatmap(observations, out_path=None):
    """Observation-density heatmap over the camera network."""
    import folium
    from folium.plugins import HeatMap

    obs = [o if isinstance(o, dict) else o.to_dict() for o in observations]
    lats = [c.latitude for c in CAMERAS.values()]
    lons = [c.longitude for c in CAMERAS.values()]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=13, tiles="CartoDB dark_matter")

    counts = Counter(o["camera_id"] for o in obs)
    if counts:
        peak = max(counts.values())
        HeatMap([[CAMERAS[c].latitude, CAMERAS[c].longitude, n / peak]
                 for c, n in counts.items() if c in CAMERAS],
                radius=45, blur=32, min_opacity=0.35).add_to(m)

    for cam_id, c in CAMERAS.items():
        folium.CircleMarker(
            [c.latitude, c.longitude], radius=9, color="#ffd166",
            fill=True, fill_opacity=0.9,
            popup=f"<b>{cam_id}</b> {c.name}<br>{counts.get(cam_id,0)} observations",
        ).add_to(m)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m
