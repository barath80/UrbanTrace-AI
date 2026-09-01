/**
 * City Traffic Analytics — the centralised, GIS-integrated view over the whole
 * camera network.
 *
 * Everything here reads the continuous `net_*` tables through /api/network, so
 * it describes the live estate rather than one batch run. Every figure is
 * measured: densities are counted sightings, speeds are derived from real
 * arrival times, and congestion is each corridor compared against its OWN
 * baseline for the same hour rather than a fixed threshold.
 *
 * Where a number cannot be computed it renders as an em dash. Nothing is
 * estimated to fill a gap.
 */
import { useCallback, useEffect, useState } from 'react'

import {
  getNetCongestion,
  getNetDensity,
  getNetFlows,
  getNetHeatmap,
  getNetSummary,
  getNetTrend,
  type Camera,
  type NetCongestion,
  type NetDensity,
  type NetFlow,
  type NetHeatPoint,
  type NetSummary,
  type NetTrendBucket,
} from '../api'
import CityMap, { type FlowLine } from '../components/CityMap'
import { Empty, Loading, Panel, Tile, num } from '../components/ui'

const WINDOWS = [
  { label: 'Last hour', hours: 1 },
  { label: 'Last 6 hours', hours: 6 },
  { label: 'Last 24 hours', hours: 24 },
  { label: 'Last 7 days', hours: 168 },
]

const SEVERITY_KIND: Record<string, string> = {
  SEVERE: 'critical',
  HEAVY: 'serious',
  MODERATE: 'warn',
}

export default function CityNetwork() {
  const [hours, setHours] = useState(24)
  const [summary, setSummary] = useState<NetSummary | null>(null)
  const [density, setDensity] = useState<NetDensity[]>([])
  const [heat, setHeat] = useState<NetHeatPoint[]>([])
  const [flows, setFlows] = useState<NetFlow[]>([])
  const [congestion, setCongestion] = useState<NetCongestion[]>([])
  const [trend, setTrend] = useState<NetTrendBucket[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback((h: number) => {
    setLoading(true)
    setError(null)
    Promise.all([
      getNetSummary(h),
      getNetDensity(h),
      getNetHeatmap(h),
      getNetFlows(h),
      getNetCongestion(h, 1),
      getNetTrend(h, h <= 6 ? 5 : h <= 24 ? 30 : 180),
    ])
      .then(([s, d, hm, f, c, t]) => {
        setSummary(s); setDensity(d); setHeat(hm)
        setFlows(f); setCongestion(c); setTrend(t)
      })
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { load(hours) }, [hours, load])

  /* --- map inputs, all measured -------------------------------------- */

  const mapCameras: Camera[] = density.map(d => ({
    camera_id: d.camera_id,
    name: d.name,
    road: d.road,
    latitude: d.latitude,
    longitude: d.longitude,
    direction: '—',
    video_file: '',
    capture_start: d.last_sighting ?? '',
    location_source: 'config',
    has_video: false,
    frames: 0,
    processing_fps: 0,
    plate_dets: d.sightings,
    vehicles_with_plate: d.unique_vehicles,
    unique_vehicles: d.unique_vehicles,
    vehicle_dets: d.sightings,
    count_mode: null,
    count_traffic: 0,
    count_presence: 0,
    count_car: 0,
    count_motorcycle: 0,
    count_bus: 0,
    count_truck: 0,
  }))

  const maxFlow = Math.max(1, ...flows.map(f => f.vehicles))
  const flowLines: FlowLine[] = flows.map(f => ({
    from: f.from_camera,
    to: f.to_camera,
    coordinates: [[f.from_lat, f.from_lon], [f.to_lat, f.to_lon]],
    weight: f.vehicles / maxFlow,
    // Straight lines between camera positions, not surveyed roads.
    real: false,
    popup: `<b>${f.from_camera} to ${f.to_camera}</b><br>${f.vehicles} vehicle(s)<br>` +
           `${f.mean_travel_minutes ?? '—'} min · ${f.mean_speed_kmh ?? '—'} km/h`,
  }))

  const peak = Math.max(1, ...trend.map(t => t.sightings))
  const busiest = density.filter(d => d.sightings > 0)

  if (loading && !summary) return <Loading what="Loading city network…" />

  return (
    <div className="net-view">
      <Panel
        title="City traffic analytics"
        hint={
          <div className="net-window">
            {WINDOWS.map(wnd => (
              <button
                key={wnd.hours}
                className={`pill${hours === wnd.hours ? ' active' : ''}`}
                onClick={() => setHours(wnd.hours)}
              >
                {wnd.label}
              </button>
            ))}
          </div>
        }
      >
        {error && <div className="mc-error">{error}</div>}

        <div className="tiles">
          <Tile label="Cameras online" value={summary?.active_cameras ?? null}
                sub={`of ${num(summary?.cameras)} in the estate`} />
          <Tile label="Sightings" value={summary?.sightings ?? null}
                sub="confirmed plate reads" accent="var(--s1)" />
          <Tile label="Unique vehicles" value={summary?.unique_vehicles ?? null}
                sub="distinct plates" accent="var(--s3)" />
          <Tile label="Multi-camera vehicles" value={summary?.multi_camera_vehicles ?? null}
                sub="tracked across the network" accent="var(--s4)" />
          <Tile label="Average speed" value={summary?.mean_speed_kmh ?? null} decimals={1}
                sub="km/h, measured between cameras" />
          <Tile label="Mean OCR confidence"
                raw={summary?.mean_confidence == null
                  ? '—' : `${(summary.mean_confidence * 100).toFixed(1)}%`}
                sub="across all sightings" />
          <Tile label="Alerts" value={summary?.alerts ?? null}
                sub="in this window"
                accent={summary && summary.alerts > 0 ? 'var(--critical)' : undefined} />
        </div>
      </Panel>

      <Panel
        title="Network map"
        hint={
          <span className="hint-text">
            Heat shows where plates were read. Lines are measured vehicle movements
            between cameras, drawn dashed because they are straight-line estimates,
            not surveyed roads.
          </span>
        }
      >
        {heat.length === 0 ? (
          <Empty title="No sightings in this window">
            Widen the window, or start ingest so the network begins recording.
          </Empty>
        ) : (
          <div className="net-map">
            <CityMap
              cameras={mapCameras}
              heat={heat}
              heatScale="count"
              flows={flowLines}
              showHeat
              showFlows={flowLines.length > 0}
              showCameras
              pitch={0}
            />
          </div>
        )}
      </Panel>

      <div className="grid-2">
        <Panel title="Route density"
               hint={<span className="hint-text">Busiest camera nodes</span>}>
          {busiest.length === 0 ? (
            <Empty title="No traffic recorded" />
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Camera</th><th>Location</th>
                    <th className="ta-r">Sightings</th>
                    <th className="ta-r">Vehicles</th>
                    <th className="ta-r">Per minute</th>
                  </tr>
                </thead>
                <tbody>
                  {busiest.slice(0, 15).map(d => (
                    <tr key={d.camera_id}>
                      <td className="mono">{d.camera_id}</td>
                      <td>{d.name}</td>
                      <td className="ta-r mono">{num(d.sightings)}</td>
                      <td className="ta-r mono">{num(d.unique_vehicles)}</td>
                      <td className="ta-r mono">{num(d.vehicles_per_minute, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel title="Traffic flow trend"
               hint={<span className="hint-text">Sightings over time across all nodes</span>}>
          {trend.length === 0 ? (
            <Empty title="No trend data yet" />
          ) : (
            <div className="net-trend">
              {trend.map(t => (
                <div key={t.bucket} className="net-bar-wrap"
                     title={`${new Date(t.bucket).toLocaleString()}\n` +
                            `${t.sightings} sighting(s), ${t.unique_vehicles} vehicle(s), ` +
                            `${t.active_cameras} camera(s)`}>
                  <div className="net-bar"
                       style={{ height: `${Math.max(3, (t.sightings / peak) * 100)}%` }} />
                </div>
              ))}
            </div>
          )}
          {trend.length > 0 && (
            <div className="net-trend-axis">
              <span>{new Date(trend[0].bucket).toLocaleString()}</span>
              <span>peak {num(peak)} / bucket</span>
              <span>{new Date(trend[trend.length - 1].bucket).toLocaleString()}</span>
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="Origin–destination"
        hint={<span className="hint-text">Measured movements between camera pairs</span>}
      >
        {flows.length === 0 ? (
          <Empty title="No completed journeys yet">
            A pair appears once the same vehicle is read at two cameras.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>From</th><th>To</th>
                  <th className="ta-r">Vehicles</th>
                  <th className="ta-r">Mean travel</th>
                  <th className="ta-r">Mean speed</th>
                  <th className="ta-r">Distance</th>
                </tr>
              </thead>
              <tbody>
                {flows.slice(0, 20).map(f => (
                  <tr key={`${f.from_camera}-${f.to_camera}`}>
                    <td><span className="mono">{f.from_camera}</span>{' '}
                        <span className="net-dim">{f.from_name}</span></td>
                    <td><span className="mono">{f.to_camera}</span>{' '}
                        <span className="net-dim">{f.to_name}</span></td>
                    <td className="ta-r mono">{num(f.vehicles)}</td>
                    <td className="ta-r mono">{num(f.mean_travel_minutes, 1)} min</td>
                    <td className="ta-r mono">{num(f.mean_speed_kmh, 1)} km/h</td>
                    <td className="ta-r mono">{num(f.road_distance_km, 2)} km</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel
        title="Congestion bottlenecks"
        hint={
          <span className="hint-text">
            Corridors running slower than their own baseline for this hour of day
          </span>
        }
      >
        {congestion.length === 0 ? (
          <Empty title="No congestion detected">
            A corridor is flagged only once it has enough history to have a normal
            speed to be compared against.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Corridor</th>
                  <th className="ta-r">Now</th>
                  <th className="ta-r">Baseline</th>
                  <th className="ta-r">Ratio</th>
                  <th className="ta-r">Vehicles</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {congestion.map(c => (
                  <tr key={`${c.from_camera}-${c.to_camera}`}>
                    <td>
                      <span className="mono">{c.from_camera} → {c.to_camera}</span>
                      <div className="net-dim">{c.from_name} to {c.to_name}</div>
                    </td>
                    <td className="ta-r mono">{num(c.speed_kmh, 1)} km/h</td>
                    <td className="ta-r mono">{num(c.baseline_kmh, 1)} km/h</td>
                    <td className="ta-r mono">{num((c.speed_ratio ?? 0) * 100, 0)}%</td>
                    <td className="ta-r mono">{num(c.vehicles)}</td>
                    <td>
                      <span className={`badge ${SEVERITY_KIND[c.severity] ?? 'warn'}`}>
                        {c.severity}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
