/** Run-level summary: what this pipeline execution actually produced. */
import { useEffect, useState } from 'react'

import {
  getCameras, getSummary, getVehicles,
  type Camera, type Summary, type VehicleRow,
} from '../api'
import { Badge, Empty, Loading, NoRun, Panel, Tile, clock, num } from '../components/ui'

export default function Overview({ onTrack }: { onTrack: (plate: string) => void }) {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [cameras, setCameras] = useState<Camera[]>([])
  const [vehicles, setVehicles] = useState<VehicleRow[]>([])
  const [busy, setBusy] = useState(true)
  const [err, setErr] = useState(false)

  useEffect(() => {
    Promise.all([getSummary(), getCameras(), getVehicles()])
      .then(([s, c, v]) => { setSummary(s); setCameras(c); setVehicles(v) })
      .catch(() => setErr(true))
      .finally(() => setBusy(false))
  }, [])

  if (busy) return <Panel><Loading /></Panel>
  if (err || !summary) return <Panel>{NoRun}</Panel>

  const counted = cameras.reduce((a, c) => a + (c.unique_vehicles || 0), 0)
  const peak = Math.max(1, ...cameras.map(c => c.vehicles_with_plate))

  return (
    <>
      <div className="tiles">
        <Tile label="Cameras" value={summary.cameras} sub="in this run" />
        <Tile label="Vehicles counted" value={counted} accent="var(--s3)"
              sub="unique, deduped across tracks" />
        <Tile label="Plates identified" value={summary.unique_plates} accent="var(--s1)"
              sub={`${summary.observations} observations`} />
        <Tile label="Multi-camera routes" value={summary.multi_camera} accent="var(--s4)"
              sub="reconstructed trajectories" />
        <Tile label="Mean OCR confidence" value={summary.mean_ocr_conf} decimals={3}
              sub={`${summary.raw_readings} raw frame reads`} />
        <Tile label="Alerts" value={summary.alerts}
              accent={summary.alerts ? 'var(--critical)' : undefined}
              sub={summary.alerts ? 'watchlist matches' : 'no watchlist hits'} />
      </div>

      <div className="grid grid-2">
        <Panel title="Per-camera activity">
          {cameras.length === 0 && <Empty title="No cameras in this run" />}
          {cameras.map(c => (
            <div key={c.camera_id} style={{ marginBottom: 15 }}>
              <div style={{ fontSize: 12.5, marginBottom: 5 }}>
                <b className="mono">{c.camera_id}</b> — {c.name}{' '}
                <span style={{ color: 'var(--ink-muted)' }}>
                  · {c.unique_vehicles} vehicles · {c.vehicles_with_plate} identified
                </span>
              </div>
              <div style={{
                height: 7, borderRadius: 4, background: 'var(--surface-3)', overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', width: `${(c.vehicles_with_plate / peak) * 100}%`,
                  background: 'linear-gradient(90deg,var(--s1),var(--s3))',
                  transition: 'width .7s cubic-bezier(.2,.8,.2,1)',
                }} />
              </div>
            </div>
          ))}
        </Panel>

        <Panel title="Reconstructed vehicles" hint="click a plate for its full dossier" flush>
          <div className="table-wrap" style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Plate</th><th>Route</th><th className="num">Cams</th>
                  <th>First seen</th><th className="num">Conf</th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map(v => (
                  <tr key={v.plate_number}>
                    <td className="plate-cell" onClick={() => onTrack(v.plate_number)}>
                      {v.plate_number}
                    </td>
                    <td className="mono" style={{ fontSize: 11.5 }}>
                      {v.route_display}{' '}
                      {v.camera_count > 1 && <Badge kind="ok">route</Badge>}
                    </td>
                    <td className="num">{v.camera_count}</td>
                    <td className="mono" style={{ fontSize: 11.5 }}>{clock(v.first_seen)}</td>
                    <td className="num">{num(v.confidence, 3)}</td>
                  </tr>
                ))}
                {vehicles.length === 0 && (
                  <tr><td colSpan={5}><Empty title="No plates identified in this run" /></td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </>
  )
}
