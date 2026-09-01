/**
 * City GIS: every camera, every corridor flow along its real road, and the
 * measured density of plate reads — with any one vehicle's route overlaid.
 */
import { useEffect, useMemo, useState } from 'react'

import {
  getCameras, getFlows, getHeatmap, getRouting, getVehiclePaths, roadRoute,
  type Camera, type Flow, type HeatPoint, type RoutingStatus, type VehiclePath,
} from '../api'
import CityMap, { type FlowLine, type RouteLine } from '../components/CityMap'
import { Loading, Panel, Provenance, num } from '../components/ui'

export default function GisMap({ onTrack }: { onTrack: (plate: string) => void }) {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [flows, setFlows] = useState<Flow[]>([])
  const [heat, setHeat] = useState<HeatPoint[]>([])
  const [paths, setPaths] = useState<Record<string, VehiclePath>>({})
  const [routing, setRouting] = useState<RoutingStatus | null>(null)
  const [roads, setRoads] = useState<Record<string, { coords: [number, number][]; real: boolean; km: number }>>({})
  const [busy, setBusy] = useState(true)

  const [active, setActive] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [showHeat, setShowHeat] = useState(true)
  const [showFlows, setShowFlows] = useState(true)
  const [showCameras, setShowCameras] = useState(true)
  const [pitch, setPitch] = useState(45)

  useEffect(() => {
    let live = true
    Promise.all([getCameras(), getFlows(), getHeatmap(), getVehiclePaths(), getRouting()])
      .then(([c, f, h, p, r]) => {
        if (!live) return
        setCameras(c); setFlows(f); setHeat(h); setPaths(p); setRouting(r)
      })
      .catch(() => { /* endpoints report emptiness themselves */ })
      .finally(() => live && setBusy(false))
    return () => { live = false }
  }, [])

  // Corridor geometry, fetched once per camera pair and cached server-side.
  useEffect(() => {
    let live = true
    ;(async () => {
      const out: typeof roads = {}
      for (const f of flows) {
        try {
          const r = await roadRoute(f.from_camera, f.to_camera)
          out[`${f.from_camera}->${f.to_camera}`] = {
            coords: r.coordinates as [number, number][],
            real: r.is_real_road,
            km: r.distance_km,
          }
        } catch { /* falls back to the straight line below */ }
      }
      if (live) setRoads(out)
    })()
    return () => { live = false }
  }, [flows])

  const byId = useMemo(
    () => Object.fromEntries(cameras.map(c => [c.camera_id, c])), [cameras])

  const flowLines: FlowLine[] = useMemo(() => {
    const max = Math.max(1, ...flows.map(f => f.vehicles))
    return flows.flatMap(f => {
      const a = byId[f.from_camera], b = byId[f.to_camera]
      if (!a || !b) return []
      const road = roads[`${f.from_camera}->${f.to_camera}`]
      const coords = road?.coords?.length
        ? road.coords
        : ([[a.latitude, a.longitude], [b.latitude, b.longitude]] as [number, number][])
      const km = road?.km ?? f.road_distance_km
      return [{
        from: f.from_camera, to: f.to_camera, coordinates: coords,
        weight: f.vehicles / max, real: !!road?.real,
        popup:
          `<b>${f.from_camera} → ${f.to_camera}</b><br>${f.vehicles} vehicle(s)<br>` +
          `${num(km, 2)} km · ${num(f.mean_travel_minutes, 1)} min measured · ` +
          `${num(f.mean_speed_kmh, 1)} km/h<br>` +
          `<em>${road?.real ? 'real road' : 'straight-line estimate'}</em>`,
      }]
    })
  }, [flows, byId, roads])

  const routed = useMemo(
    () => Object.values(paths).filter(p => p.route.length > 1), [paths])

  const listed = useMemo(
    () => routed.filter(p => p.plate_number.toUpperCase().includes(filter.toUpperCase())),
    [routed, filter])

  const route: RouteLine | null = useMemo(() => {
    const p = active ? paths[active] : null
    if (!p) return null
    const coords = p.coordinates.length > 1
      ? p.coordinates
      : p.points.map(s => [s.latitude, s.longitude] as [number, number])
    return {
      coordinates: coords,
      real: p.is_real_road,
      stops: p.points.map((s, i) => ({
        lat: s.latitude, lon: s.longitude, label: String(i + 1),
        popup: `<b>${p.plate_number}</b> — stop ${i + 1} of ${p.points.length}<br>` +
               `${s.camera_id} · ${s.name}`,
      })),
    }
  }, [active, paths])

  if (busy) return <Panel><Loading what="Loading the camera network…" /></Panel>

  const sel = active ? paths[active] : null

  return (
    <Panel
      title="City GIS — real road routes, cameras and traffic density"
      hint={`${cameras.length} cameras · ${flows.length} flows · ${routed.length} routes`}
      flush
    >
      <div className="map-shell">
        <aside className="map-side">
          <div className="map-block">
            <h3>Vehicle route</h3>
            <p className="map-note" style={{ marginBottom: 10 }}>
              Pick a vehicle to draw the road it travelled between the cameras that saw it.
            </p>
            {routed.length > 0 && (
              <input
                className="input sm" placeholder="Filter by plate…"
                value={filter} onChange={e => setFilter(e.target.value)}
                style={{ marginBottom: 9 }}
              />
            )}
            <div className="route-list">
              {routed.length === 0 && (
                <p className="map-note">
                  No multi-camera routes in this run. A route needs the same plate read at
                  two or more cameras.
                </p>
              )}
              {routed.length > 0 && listed.length === 0 && (
                <p className="map-note">No plate matches that filter.</p>
              )}
              {listed.map(p => (
                <button
                  key={p.global_vehicle_id}
                  className={`route-item${active === p.global_vehicle_id ? ' on' : ''}`}
                  onClick={() => setActive(
                    active === p.global_vehicle_id ? null : p.global_vehicle_id)}
                >
                  <div className="rp">{p.plate_number}</div>
                  <div className="rr">
                    {p.route.join(' → ')} · {num(p.road_distance_km, 2)} km
                    {p.is_real_road ? '' : ' · estimated'}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {sel && (
            <div className="map-block">
              <h3>Selected route</h3>
              <div className="fact"><span>Plate</span><span>{sel.plate_number}</span></div>
              <div className="fact"><span>Route</span><span>{sel.route.join(' → ')}</span></div>
              <div className="fact">
                <span>Road distance</span><span>{num(sel.road_distance_km, 2)} km</span>
              </div>
              <div className="fact">
                <span>Measured travel</span><span>{num(sel.measured_duration_min, 1)} min</span>
              </div>
              <div className="fact">
                <span>Typical for road</span><span>{num(sel.typical_duration_min, 1)} min</span>
              </div>
              <div className="fact">
                <span>Path source</span><span><Provenance real={sel.is_real_road} /></span>
              </div>
              <p className="map-note" style={{ marginTop: 10 }}>{sel.path_note}</p>
              <div style={{ display: 'flex', gap: 8, marginTop: 11, flexWrap: 'wrap' }}>
                <button className="btn sm" onClick={() => onTrack(sel.plate_number)}>
                  Full dossier
                </button>
                <button className="btn ghost sm" onClick={() => setActive(null)}>
                  Clear
                </button>
              </div>
            </div>
          )}

          <div className="map-block">
            <h3>Layers</h3>
            <label className="chk">
              <input type="checkbox" checked={showHeat}
                     onChange={e => setShowHeat(e.target.checked)} />
              <span>Traffic density heatmap</span>
            </label>
            <label className="chk">
              <input type="checkbox" checked={showFlows}
                     onChange={e => setShowFlows(e.target.checked)} />
              <span>Corridor flows (real roads)</span>
            </label>
            <label className="chk">
              <input type="checkbox" checked={showCameras}
                     onChange={e => setShowCameras(e.target.checked)} />
              <span>Camera nodes</span>
            </label>
            <label className="chk" style={{ marginTop: 8 }}>
              <input type="checkbox" checked={pitch > 10}
                     onChange={e => setPitch(e.target.checked ? 50 : 0)} />
              <span>3D view (tilt + buildings)</span>
            </label>
            {routing && (
              <p className="map-note" style={{ marginTop: 11 }}>
                <Provenance real={routing.api_key_configured}
                            realLabel="real roads" estLabel="estimates" />{' '}
                {routing.api_key_configured
                  ? `GraphHopper · ${routing.cached_routes} cached`
                  : 'No GRAPHHOPPER_API_KEY in .env — paths fall back to straight lines.'}
              </p>
            )}
          </div>
        </aside>

        <CityMap
          cameras={cameras} heat={heat} heatScale="count"
          flows={flowLines} route={route}
          showHeat={showHeat} showFlows={showFlows} showCameras={showCameras}
          pitch={pitch} className="map-canvas tall"
        />
      </div>

      <div className="legend">
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--s1)' }} />Camera node
        </span>
        <span className="legend-item">
          <span className="swatch line" style={{ background: 'var(--s1)' }} />
          Corridor flow (thickness = volume)
        </span>
        <span className="legend-item">
          <span className="swatch line" style={{ background: 'var(--s2)' }} />
          Selected vehicle route
        </span>
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--s3)' }} />Low density
        </span>
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--warning)' }} />Moderate
        </span>
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--critical)' }} />High
        </span>
        <span className="legend-item" style={{ color: 'var(--ink-muted)' }}>
          Dashed = straight-line estimate, not a surveyed road
        </span>
      </div>
    </Panel>
  )
}
