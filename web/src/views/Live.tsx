/**
 * Real-time monitoring.
 *
 * Subscribes to /api/live/stream (server-sent events) and renders what the
 * detector is finding right now: the frame, the running counts, plates as they
 * reach consensus, and watchlist alerts the moment they fire.
 *
 * Nothing here is simulated. If no session is running the view says so and
 * offers to start one; it never shows placeholder activity.
 */
import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useRef, useState } from 'react'

import { api, getCameras, liveStart, liveStop, type Camera } from '../api'
import { Badge, Empty, Panel, Tile, num } from '../components/ui'

interface LiveStatus {
  camera_id: string
  name: string
  source: string
  state: string
  error: string | null
  uptime_s: number
  frames: number
  fps: number
  source_fps: number
  realtime_factor: number | null
  keeping_up: boolean | null
  detect_every: number
  cycles: number
  detections: number
  plates_read: number
  alerts: number
  counts: Record<string, number>
  watchlist_size: number
}

interface PlateEvent {
  at: string
  plate_number: string
  ocr_confidence: number
  support: number
  total_readings: number
  agreement: number
  vehicle_type: string
  location: string
  camera_id: string
  evidence_b64?: string | null
  severity?: string
  reason?: string
}

const MAX_FEED = 40

export default function Live({ onTrack }: { onTrack: (plate: string) => void }) {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [selected, setSelected] = useState<string>('')
  const [detectEvery, setDetectEvery] = useState(3)
  const [plates, setPlates] = useState<PlateEvent[]>([])
  const [alerts, setAlerts] = useState<PlateEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [frameTick, setFrameTick] = useState(0)

  const es = useRef<EventSource | null>(null)

  useEffect(() => {
    getCameras()
      .then(c => { setCameras(c); if (c.length && !selected) setSelected(c[0].camera_id) })
      .catch(() => setCameras([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connect = useCallback((cam: string) => {
    es.current?.close()
    const src = new EventSource(`/api/live/stream?camera_id=${encodeURIComponent(cam)}`)
    es.current = src

    src.addEventListener('hello', e =>
      { setStatus(JSON.parse((e as MessageEvent).data)); setConnected(true) })
    src.addEventListener('started', e => setStatus(JSON.parse((e as MessageEvent).data)))
    src.addEventListener('stats', e => setStatus(JSON.parse((e as MessageEvent).data)))
    src.addEventListener('cycle', e => setStatus(JSON.parse((e as MessageEvent).data)))
    src.addEventListener('stopped', e => {
      setStatus(JSON.parse((e as MessageEvent).data)); setConnected(false); src.close()
    })
    src.addEventListener('plate', e => {
      const p = JSON.parse((e as MessageEvent).data) as PlateEvent
      setPlates(prev => [p, ...prev].slice(0, MAX_FEED))
    })
    src.addEventListener('alert', e => {
      const a = JSON.parse((e as MessageEvent).data) as PlateEvent
      setAlerts(prev => [a, ...prev].slice(0, MAX_FEED))
    })
    src.onerror = () => setConnected(false)
  }, [])

  // Attach to a session that is already running - after a page reload, or on a
  // second device. Showing its status without subscribing left the view frozen
  // on a stale snapshot that read "reconnecting" and zero vehicles.
  useEffect(() => {
    api<{ sessions: LiveStatus[] }>('/live/status')
      .then(s => {
        const run = s.sessions.find(x => x.state === 'running')
        if (!run) return
        setSelected(run.camera_id)
        setStatus(run)
        setDetectEvery(run.detect_every || 3)
        connect(run.camera_id)
      })
      .catch(() => { /* nothing running */ })
  }, [connect])


  useEffect(() => () => es.current?.close(), [])

  // Poll the preview frame while a session runs. An <img> with a cache-buster
  // is enough here and avoids holding a second long-lived connection open.
  useEffect(() => {
    if (status?.state !== 'running' || !status.frames) return
    const t = setInterval(() => setFrameTick(n => n + 1), 400)
    return () => clearInterval(t)
  }, [status?.state, status?.frames])

  const start = async () => {
    if (!selected) return
    setBusy(true); setErr(null); setPlates([]); setAlerts([])
    try {
      const s = await liveStart(selected, detectEvery) as LiveStatus
      setStatus(s)
      connect(selected)
    } catch (e) {
      setErr(String(e))
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      await liveStop(selected)
      es.current?.close()
      setConnected(false)
    } finally {
      setBusy(false)
    }
  }

  const running = status?.state === 'running'
  const counts = status?.counts ?? {}

  return (
    <>
      <div className="note">
        Live detection runs the same vision stack continuously against a source
        OpenCV can open — an <code className="mono">rtsp://</code> camera, a local
        device index, or a video file played at its own frame rate. Plates are
        published only once temporal consensus is reached, not on a single frame.
      </div>

      <Panel title="Feed control">
        <div className="control-row">
          <label className="field">
            <span>Camera</span>
            <select className="input sm" value={selected}
                    onChange={e => setSelected(e.target.value)} disabled={running}>
              {cameras.map(c => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id} — {c.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Detect every Nth frame</span>
            <select className="input sm" value={detectEvery}
                    onChange={e => setDetectEvery(Number(e.target.value))}
                    disabled={running}>
              {[1, 2, 3, 4, 6, 8].map(n => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          <div className="field grow">
            <span>&nbsp;</span>
            <div style={{ display: 'flex', gap: 8 }}>
              {!running
                ? <button className="btn" onClick={start} disabled={busy || !selected}>
                    {busy ? 'Starting…' : 'Start live detection'}
                  </button>
                : <button className="btn ghost" onClick={stop} disabled={busy}>
                    Stop
                  </button>}
              {running && (
                <span className="chip">
                  <span className={`dot ${connected ? 'ok live' : 'bad'}`} />
                  {connected ? 'streaming' : 'reconnecting…'}
                </span>
              )}
            </div>
          </div>
        </div>
        {err && <p className="map-note" style={{ color: 'var(--critical)' }}>{err}</p>}
        {status?.error && (
          <p className="map-note" style={{ color: 'var(--critical)' }}>{status.error}</p>
        )}
      </Panel>

      {!running && !status && (
        <Panel>
          <Empty title="No live session running">
            Start one above. With no source attached it plays the camera's own
            video file at its real frame rate, so the feed behaves like a live one.
          </Empty>
        </Panel>
      )}

      {status && (
        <>
          <div className="tiles">
            <Tile label="Vehicles (live)" value={counts.total ?? 0} accent="var(--s3)"
                  sub="counted this cycle" />
            <Tile label="Plates read" value={status.plates_read} accent="var(--s1)"
                  sub="temporal consensus reached" />
            <Tile label="Alerts" value={status.alerts}
                  accent={status.alerts ? 'var(--critical)' : undefined}
                  sub={`${status.watchlist_size} on watchlist`} />
            <Tile label="Throughput" value={status.fps} decimals={1}
                  accent={status.keeping_up === false ? 'var(--warning)' : 'var(--s4)'}
                  sub={<>
                    fps · source {num(status.source_fps, 0)}{' '}
                    {status.keeping_up === false
                      ? <Badge kind="warn">behind</Badge>
                      : <Badge kind="ok">real time</Badge>}
                  </>} />
            <Tile label="Uptime" value={status.uptime_s} decimals={0}
                  sub={`seconds · ${status.frames} frames · cycle ${status.cycles + 1}`} />
          </div>

          <div className="grid grid-2">
            <Panel
              title="Live feed"
              hint={running
                ? `${status.camera_id} · detecting every ${status.detect_every} frame(s)`
                : 'stopped'}
              flush
            >
              {running && status.frames > 0 ? (
                <img
                  className="live-frame"
                  alt="live camera frame"
                  src={`/api/live/frame/${status.camera_id}?t=${frameTick}`}
                />
              ) : running ? (
                <Empty title="Waiting for the first frame…" />
              ) : (
                <Empty title="Feed stopped" />
              )}
              <div className="legend">
                <span className="legend-item">
                  Source: <code className="mono">{status.source.split(/[\\/]/).pop()}</code>
                </span>
                {status.keeping_up === false && (
                  <span className="legend-item" style={{ color: 'var(--warning)' }}>
                    Detection is slower than the feed — raise “detect every Nth frame”.
                  </span>
                )}
              </div>
            </Panel>

            <Panel title="Live plate reads" hint={`${plates.length} most recent`} flush>
              <div className="feed">
                <AnimatePresence initial={false}>
                  {plates.map((p, i) => (
                    <motion.div
                      className="feed-row" key={`${p.plate_number}-${p.at}-${i}`}
                      initial={{ opacity: 0, x: -14 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.22 }}
                    >
                      {p.evidence_b64 && (
                        <img alt="" src={`data:image/jpeg;base64,${p.evidence_b64}`} />
                      )}
                      <div className="feed-body">
                        <div className="feed-top">
                          <button className="plate-cell"
                                  onClick={() => onTrack(p.plate_number)}>
                            {p.plate_number}
                          </button>
                          <Badge kind={p.ocr_confidence >= 0.85 ? 'ok' : 'warn'}>
                            {num(p.ocr_confidence, 3)}
                          </Badge>
                          <span className="feed-meta">
                            {p.support}/{p.total_readings} frames agree
                          </span>
                        </div>
                        <div className="feed-meta">
                          {p.camera_id} · {p.location} · {p.vehicle_type} ·{' '}
                          {p.at.slice(11, 19)}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {plates.length === 0 && (
                  <Empty title="No plates read yet">
                    A plate appears here once enough frames agree on it.
                  </Empty>
                )}
              </div>
            </Panel>
          </div>

          <Panel title="Live watchlist alerts" hint={`${alerts.length} this session`}>
            {alerts.length === 0 ? (
              <Empty title="No live alerts">
                {status.watchlist_size === 0
                  ? 'The watchlist is empty, so nothing can match.'
                  : 'No watchlisted plate has been seen since this session started.'}
              </Empty>
            ) : (
              <AnimatePresence initial={false}>
                {alerts.map((a, i) => (
                  <motion.div
                    className={`alert-card ${a.severity ?? 'MEDIUM'}`}
                    key={`${a.plate_number}-${a.at}-${i}`}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.24 }}
                  >
                    {a.evidence_b64 && (
                      <img alt="" src={`data:image/jpeg;base64,${a.evidence_b64}`} />
                    )}
                    <div className="body">
                      <div className="t">
                        <Badge kind={a.severity === 'HIGH' ? 'critical' : 'warn'}>
                          {a.severity}
                        </Badge>
                        <Badge kind="info">live watchlist match</Badge>
                        <button className="plate-cell"
                                onClick={() => onTrack(a.plate_number)}>
                          {a.plate_number}
                        </button>
                        <span className="feed-meta">{a.at.slice(11, 19)}</span>
                      </div>
                      <div className="r">{a.reason}</div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </Panel>
        </>
      )}
    </>
  )
}
