/**
 * Multi-camera monitoring: three phone feeds, one network.
 *
 * Each phone streams a stored video to its own camera slot. The backend runs
 * the same vehicle tracker, plate detector and OCR on every feed, then a
 * central registry links plates seen by more than one camera into a single
 * journey.
 *
 * Every figure on this page is measured and comes from the API. Counts are
 * real counts, sighting positions are the configured camera positions, and the
 * lines drawn between cameras are straight-line estimates that say so. Nothing
 * here is filled in to make the page look busier than the data supports.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  connectMultiCamLocal,
  connectMultiCamUrl,
  getMultiCamInfo,
  getMultiCamStatus,
  getMultiCamVehicles,
  resetMultiCamRegistry,
  stopAllMultiCamStreams,
  stopMultiCamStream,
  type Camera,
  type MultiCamDetection,
  type MultiCamInfo,
  type MultiCamNetwork,
  type MultiCamSlot,
  type MultiCamVehicle,
} from '../api'
import CityMap, { type FlowLine } from '../components/CityMap'
import QRCode from '../components/QRCode'
import { Empty, Panel, num, verdictKind } from '../components/ui'

const CAMERA_IDS = ['CAM_001', 'CAM_002', 'CAM_003']

const STATUS_TEXT: Record<string, string> = {
  playing: 'Streaming',
  reconnecting: 'Reconnecting',
  disconnected: 'Disconnected',
  stopped: 'Stopped',
  waiting: 'Waiting for a phone',
  error: 'Error',
  ended: 'Ended',
}

const STATUS_DOT: Record<string, string> = {
  playing: 'ok',
  reconnecting: 'warn',
  disconnected: 'bad',
  stopped: 'muted',
  waiting: 'muted',
  error: 'bad',
  ended: 'muted',
}

/** Local time of an ISO instant, or an em dash when there is nothing to show. */
function localTime(iso?: string | null): string {
  if (!iso) return '—'
  const t = new Date(iso)
  return Number.isNaN(t.getTime()) ? '—' : t.toLocaleTimeString()
}

export default function MultiCamera({ onTrack }: { onTrack?: (plate: string) => void } = {}) {
  const [slots, setSlots] = useState<MultiCamSlot[]>([])
  const [detections, setDetections] = useState<MultiCamDetection[]>([])
  const [vehicles, setVehicles] = useState<MultiCamVehicle[]>([])
  const [network, setNetwork] = useState<MultiCamNetwork | null>(null)
  const [info, setInfo] = useState<MultiCamInfo | null>(null)
  const [showSetup, setShowSetup] = useState(false)
  const [urlCam, setUrlCam] = useState('CAM_001')
  const [inputUrl, setInputUrl] = useState('')
  const [zoomCam, setZoomCam] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)

  const gridRef = useRef<HTMLDivElement>(null)

  const refresh = useCallback(() => {
    getMultiCamStatus()
      .then(res => {
        const byId = new Map(res.cameras.map(c => [c.camera_id, c]))
        setSlots(CAMERA_IDS.map(id => byId.get(id)).filter(Boolean) as MultiCamSlot[])
        setDetections(res.detections ?? [])
        setNetwork(res.network ?? null)
      })
      .catch(() => {})
    getMultiCamVehicles().then(setVehicles).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    getMultiCamInfo().then(setInfo).catch(() => {})
    const t = setInterval(refresh, 1500)
    return () => clearInterval(t)
  }, [refresh])

  /** Wrap an action so a failure is shown in the page, not in an alert box. */
  const run = useCallback(async (label: string, fn: () => Promise<unknown>) => {
    setBusy(true)
    setError(null)
    try {
      await fn()
      refresh()
    } catch (e) {
      setError(`${label} failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const copy = (text: string, key: string) => {
    navigator.clipboard?.writeText(text).then(
      () => { setCopied(key); setTimeout(() => setCopied(null), 1500) },
      () => setError('Could not copy to the clipboard'),
    )
  }

  const phoneUrl = (camId: string) =>
    info?.phone_urls?.[camId] ??
    `http://${info?.primary_ip ?? 'localhost'}:${info?.port ?? 8000}/phone-streamer?cam=${camId}`

  const streaming = slots.filter(s => s.status === 'playing').length

  /* --- map inputs, all derived from real sightings ------------------- */

  const sightingsPerCamera = new Map<string, number>()
  for (const v of vehicles) {
    for (const s of v.sightings) {
      sightingsPerCamera.set(s.camera_id, (sightingsPerCamera.get(s.camera_id) ?? 0) + 1)
    }
  }

  const mapCameras: Camera[] = slots.map(s => ({
    camera_id: s.camera_id,
    name: s.name,
    road: s.road,
    latitude: s.latitude,
    longitude: s.longitude,
    direction: '—',
    video_file: s.video_name,
    capture_start: s.timestamp,
    location_source: 'config',
    has_video: s.has_feed,
    frames: s.frame_count,
    processing_fps: s.fps,
    plate_dets: s.plates_detected_count ?? 0,
    vehicles_with_plate: sightingsPerCamera.get(s.camera_id) ?? 0,
    unique_vehicles: sightingsPerCamera.get(s.camera_id) ?? 0,
    // Per-class totals are not tracked by the live slots, so they are reported
    // as zero rather than guessed from the frame count.
    vehicle_dets: 0,
    count_mode: null,
    count_traffic: 0,
    count_presence: 0,
    count_car: 0,
    count_motorcycle: 0,
    count_bus: 0,
    count_truck: 0,
  }))

  const heat: [number, number, number][] = slots
    .map(s => [s.latitude, s.longitude, sightingsPerCamera.get(s.camera_id) ?? 0] as [number, number, number])
    .filter(h => h[2] > 0)

  const position = new Map(slots.map(s => [s.camera_id, [s.latitude, s.longitude] as [number, number]]))

  /* One line per camera pair a vehicle was actually observed crossing. These
     are straight lines between camera positions, not surveyed roads, so they
     are marked `real: false` and the map draws them dashed. */
  const flowCounts = new Map<string, { from: string; to: string; n: number; km: number }>()
  for (const v of vehicles) {
    for (const l of v.legs) {
      const key = `${l.from_camera}->${l.to_camera}`
      const cur = flowCounts.get(key)
      if (cur) cur.n += 1
      else flowCounts.set(key, { from: l.from_camera, to: l.to_camera, n: 1, km: l.road_distance_km })
    }
  }
  const maxFlow = Math.max(1, ...[...flowCounts.values()].map(f => f.n))
  const flows: FlowLine[] = [...flowCounts.values()]
    .filter(f => position.has(f.from) && position.has(f.to))
    .map(f => ({
      from: f.from,
      to: f.to,
      coordinates: [position.get(f.from)!, position.get(f.to)!],
      weight: f.n / maxFlow,
      real: false,
      popup: `<b>${f.from} to ${f.to}</b><br>${f.n} vehicle(s)<br>${f.km.toFixed(2)} km (straight-line estimate)`,
    }))

  const multiCam = vehicles.filter(v => v.camera_count > 1)

  return (
    <div className="multicam-view">
      <Panel
        title="Multi-camera monitoring"
        hint={
          <span className="hint-text">
            {streaming} of {slots.length || 3} feeds streaming
            {network ? ` · ${network.vehicles} vehicle(s) seen · ${network.multi_camera_vehicles} on more than one camera` : ''}
          </span>
        }
      >
        <div className="mc-toolbar">
          <button className="btn btn-sm" onClick={() => setShowSetup(v => !v)}>
            {showSetup ? 'Hide phone setup' : 'Phone setup'}
          </button>
          <button
            className="btn btn-sm ghost"
            disabled={busy}
            onClick={() => run('Start all feeds', () =>
              Promise.all(CAMERA_IDS.map(id => connectMultiCamLocal(id))))}
          >
            Start all from local video
          </button>
          <button
            className="btn btn-sm ghost"
            disabled={busy}
            onClick={() => run('Stop all feeds', stopAllMultiCamStreams)}
          >
            Stop all
          </button>
          <button
            className="btn btn-sm ghost"
            disabled={busy}
            onClick={() => run('Clear history', resetMultiCamRegistry)}
            title="Clear the recorded sightings and journeys. Streams keep running."
          >
            Clear sighting history
          </button>
          <button
            className="btn btn-sm ghost"
            onClick={() => {
              if (document.fullscreenElement) document.exitFullscreen?.()
              else gridRef.current?.requestFullscreen?.()
            }}
          >
            Fullscreen grid
          </button>
        </div>

        {error && <div className="mc-error">{error}</div>}
      </Panel>

      {showSetup && (
        <Panel title="Connecting the three phones">
          <ol className="mc-steps">
            <li>Put the three phones and this laptop on the same Wi-Fi network or laptop hotspot.</li>
            <li>
              Point each phone camera at one of the codes below, or type the address into the
              phone browser — one code per phone.
            </li>
            <li>On each phone, choose a stored video file. It plays in a loop and streams here until you stop it.</li>
          </ol>

          <div className="mc-url-list">
            {CAMERA_IDS.map((id, i) => (
              <div key={id} className="mc-url-row">
                <QRCode
                  value={phoneUrl(id)}
                  size={116}
                  label={`Scan to open the Phone ${i + 1} streamer for ${id}`}
                  className="mc-qr"
                />
                <div className="mc-url-body">
                  <span className="mc-url-label">Phone {i + 1} · {id}</span>
                  <code className="mono mc-url">{phoneUrl(id)}</code>
                </div>
                <button className="btn btn-sm ghost" onClick={() => copy(phoneUrl(id), id)}>
                  {copied === id ? 'Copied' : 'Copy'}
                </button>
              </div>
            ))}
          </div>

          {info && info.local_ips.length > 1 && (
            <p className="mc-note">
              This laptop has more than one address ({info.local_ips.join(', ')}). If a phone cannot
              reach the first one, try another.
            </p>
          )}

          <div className="mc-url-form">
            <label className="mc-url-label" htmlFor="mc-stream-url">
              Or connect a network stream (RTSP or HTTP video) to
            </label>
            <select
              id="mc-stream-cam"
              className="input sm"
              value={urlCam}
              onChange={e => setUrlCam(e.target.value)}
            >
              {CAMERA_IDS.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
            <input
              id="mc-stream-url"
              className="input sm"
              placeholder="rtsp://… or http://…/video.mp4"
              value={inputUrl}
              onChange={e => setInputUrl(e.target.value)}
            />
            <button
              className="btn btn-sm"
              disabled={!inputUrl || busy}
              onClick={() => run('Connect stream', async () => {
                await connectMultiCamUrl(urlCam, inputUrl)
                setInputUrl('')
              })}
            >
              Connect
            </button>
          </div>
        </Panel>
      )}

      <div ref={gridRef} className="mc-grid">
        {slots.length === 0 && (
          <Empty title="No camera slots reported">
            The backend is not returning camera status. Check that the API is running.
          </Empty>
        )}

        {slots.map((cam, i) => {
          const playing = cam.status === 'playing'
          const zoomed = zoomCam === cam.camera_id
          return (
            <section key={cam.camera_id} className={`mc-card${zoomed ? ' is-zoomed' : ''}`}>
              <header className="mc-card-head">
                <div>
                  <span className="mc-cam-id mono">{cam.camera_id}</span>
                  <span className="mc-cam-name">{cam.name}</span>
                </div>
                <span className="mc-status">
                  <span className={`dot ${STATUS_DOT[cam.status] ?? 'muted'}`} />
                  {STATUS_TEXT[cam.status] ?? cam.status}
                </span>
              </header>

              <div className="mc-screen">
                {playing ? (
                  <img
                    className="mc-video"
                    src={`/api/multi-camera/stream/${cam.camera_id}`}
                    alt={`Live feed from ${cam.camera_id}`}
                  />
                ) : (
                  <div className="mc-screen-idle">
                    <p className="mc-idle-title">{STATUS_TEXT[cam.status] ?? cam.status}</p>
                    <p className="mc-idle-sub">
                      {cam.error_message ?? `Scan this code with Phone ${i + 1} to stream a video from it, or start from a local video.`}
                    </p>
                    <QRCode
                      value={phoneUrl(cam.camera_id)}
                      size={124}
                      label={`Scan to open the Phone ${i + 1} streamer for ${cam.camera_id}`}
                      className="mc-idle-qr"
                    />
                    <button
                      className="btn btn-sm"
                      disabled={busy}
                      onClick={() => run(`Start ${cam.camera_id}`, () => connectMultiCamLocal(cam.camera_id))}
                    >
                      Start from local video
                    </button>
                  </div>
                )}
              </div>

              <dl className="mc-meta">
                <div><dt>Source</dt><dd>{cam.video_name || '—'}</dd></div>
                <div><dt>Rate</dt><dd className="mono">{playing ? `${cam.fps.toFixed(1)} fps` : '—'}</dd></div>
                <div><dt>Frames</dt><dd className="mono">{num(cam.frame_count)}</dd></div>
                <div><dt>Plates</dt><dd className="mono">{num(cam.plates_detected_count ?? 0)}</dd></div>
              </dl>

              {(cam.detect_failures ?? 0) > 0 && (
                <p className="mc-detect-warn">
                  Detection failed {num(cam.detect_failures ?? 0)}× on this feed
                  {cam.last_detect_error ? ` — ${cam.last_detect_error}` : ''}. Plate boxes
                  on this camera are incomplete.
                </p>
              )}

              <footer className="mc-card-foot">
                <button
                  className="btn btn-sm ghost"
                  /* Always clickable: a feed reporting a stale "playing" status
                     is exactly when an operator most needs to stop it. */
                  disabled={busy}
                  onClick={() => run(`Stop ${cam.camera_id}`, () => stopMultiCamStream(cam.camera_id))}
                >
                  Stop
                </button>
                <button
                  className="btn btn-sm ghost"
                  disabled={busy}
                  onClick={() => run(`Start ${cam.camera_id}`, () => connectMultiCamLocal(cam.camera_id))}
                >
                  Start
                </button>
                <button
                  className="btn btn-sm ghost"
                  onClick={() => setZoomCam(zoomed ? null : cam.camera_id)}
                >
                  {zoomed ? 'Shrink' : 'Enlarge'}
                </button>
              </footer>
            </section>
          )
        })}
      </div>

      <Panel
        title="Vehicles seen across the network"
        hint={
          <span className="hint-text">
            {multiCam.length > 0
              ? `${multiCam.length} vehicle(s) seen by more than one camera`
              : 'A journey appears once the same plate is read at two cameras'}
          </span>
        }
      >
        {vehicles.length === 0 ? (
          <Empty title="No vehicles recorded yet">
            Plates appear here once a reading has been confirmed across several frames.
          </Empty>
        ) : (
          <div className="mc-journeys">
            {vehicles.map(v => (
              <article key={v.plate} className="mc-journey">
                <div className="mc-journey-head">
                  <button className="plate-badge-btn mono" onClick={() => onTrack?.(v.plate)}>
                    {v.plate}
                  </button>
                  <span className={`badge ${v.camera_count > 1 ? 'ok' : 'muted'}`}>
                    {v.camera_count} camera{v.camera_count === 1 ? '' : 's'}
                  </span>
                  <span className="mc-journey-meta mono">
                    {(v.confidence * 100).toFixed(0)}% mean OCR · {v.sighting_count} sighting(s)
                  </span>
                  {v.variants.length > 0 && (
                    <span className="mc-journey-meta">
                      merged reading{v.variants.length === 1 ? '' : 's'}:{' '}
                      <span className="mono">{v.variants.join(', ')}</span>
                    </span>
                  )}
                </div>

                <div className="mc-route mono">{v.route_display}</div>

                {v.legs.length > 0 && (
                  <ul className="mc-legs">
                    {v.legs.map((l, i) => (
                      <li key={i}>
                        <span className="mono">{l.from_camera} to {l.to_camera}</span>
                        <span className="mono">
                          {num(l.travel_minutes, 1)} min · {num(l.road_distance_km, 2)} km ·{' '}
                          {l.implied_speed_kmh === null ? '—' : `${num(l.implied_speed_kmh, 0)} km/h`}
                        </span>
                        <span className={`badge ${verdictKind(l.verdict)}`}>{l.verdict}</span>
                        <span className="mc-leg-note">{l.distance_source}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </article>
            ))}
          </div>
        )}
      </Panel>

      <Panel
        title="Recent plate readings"
        hint={<span className="hint-text">Newest first, across all three cameras</span>}
      >
        {detections.length === 0 ? (
          <Empty title="No readings yet">
            A reading is published once several frames agree on the same plate.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Plate</th><th>Camera</th><th>Type</th>
                  <th className="ta-r">OCR</th><th className="ta-r">Frames agreeing</th>
                  <th>Format</th><th>Seen at</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={`${d.camera_id}-${d.track_id}-${i}`}>
                    <td>
                      <button className="plate-badge-btn mono" onClick={() => onTrack?.(d.plate)}>
                        {d.plate}
                      </button>
                    </td>
                    <td className="mono">{d.camera_id}</td>
                    <td>{d.vehicle_type}</td>
                    <td className="ta-r mono">{(d.confidence * 100).toFixed(0)}%</td>
                    <td className="ta-r mono">{d.support}/{d.total_readings}</td>
                    <td>
                      <span className={`badge ${d.is_valid_format ? 'ok' : 'warn'}`}>
                        {d.is_valid_format ? 'valid' : 'unrecognised'}
                      </span>
                    </td>
                    <td className="mono">{localTime(d.observed_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel
        title="Camera network"
        hint={
          <span className="hint-text">
            Circles show where plates were read. Dashed lines are straight-line
            estimates between cameras, not surveyed roads.
          </span>
        }
      >
        <div className="mc-map">
          <CityMap
            cameras={mapCameras}
            heat={heat}
            heatScale="count"
            flows={flows}
            showHeat={heat.length > 0}
            showFlows={flows.length > 0}
            showCameras
            pitch={0}
          />
        </div>
      </Panel>
    </div>
  )
}
