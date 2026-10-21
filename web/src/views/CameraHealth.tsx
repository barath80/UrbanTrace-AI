/**
 * Camera health: which feeds are actually working, and which are quietly broken.
 *
 * A camera that is "connected" is not the same as a camera that is doing its
 * job. A feed can be streaming at full rate while the detector fails on every
 * frame, or while the GPU is so far behind that plates pass unread. Both look
 * fine on the multi-camera grid. This view names the difference.
 *
 * Every verdict here is derived from counters the backend measures - frames
 * received, detections run, detections skipped, detection failures, plates
 * published. The rule behind each verdict is printed next to it, so a DEGRADED
 * badge can always be traced back to the number that caused it rather than
 * being taken on trust.
 */
import { useCallback, useEffect, useState } from 'react'

import {
  getCameras,
  getHealth,
  getMultiCamStatus,
  type Camera,
  type Health,
  type MultiCamSlot,
} from '../api'
import { Empty, Panel, num } from '../components/ui'

type Verdict = 'healthy' | 'degraded' | 'down' | 'idle'

const VERDICT_LABEL: Record<Verdict, string> = {
  healthy: 'Healthy',
  degraded: 'Degraded',
  down: 'Down',
  idle: 'Idle',
}

const VERDICT_KIND: Record<Verdict, string> = {
  healthy: 'ok',
  degraded: 'warn',
  down: 'bad',
  idle: 'muted',
}

interface Assessment {
  verdict: Verdict
  reasons: string[]
  detectionRate: number | null   // share of received frames the detector ran on
  plateYield: number | null      // plates published per 1000 frames
}

/**
 * Judge one feed. The thresholds are deliberately explicit and reported with
 * the result - a verdict nobody can explain is worse than no verdict.
 */
function assess(cam: MultiCamSlot): Assessment {
  const frames = cam.frame_count ?? 0
  const run = cam.detections_run ?? 0
  const skipped = cam.detections_skipped ?? 0
  const attempted = run + skipped
  const detectionRate = attempted > 0 ? run / attempted : null
  const plateYield = frames > 0 ? ((cam.plates_detected_count ?? 0) / frames) * 1000 : null

  const reasons: string[] = []
  let verdict: Verdict = 'healthy'

  if (cam.status === 'stopped' || cam.status === 'waiting' || cam.status === 'ended') {
    return { verdict: 'idle', reasons: [`Feed is ${cam.status}, so nothing is being measured.`],
             detectionRate, plateYield }
  }
  if (cam.status === 'error' || cam.status === 'disconnected') {
    return {
      verdict: 'down',
      reasons: [cam.error_message || `Feed reports status "${cam.status}".`],
      detectionRate, plateYield,
    }
  }

  if ((cam.detect_failures ?? 0) > 0) {
    verdict = 'degraded'
    reasons.push(
      `Detection threw ${num(cam.detect_failures ?? 0)} time(s)` +
      `${cam.last_detect_error ? ` — ${cam.last_detect_error}` : ''}. Those frames were never analysed.`,
    )
  }
  if (detectionRate !== null && detectionRate < 0.5 && attempted > 30) {
    verdict = 'degraded'
    reasons.push(
      `The detector kept up with only ${(detectionRate * 100).toFixed(0)}% of frames ` +
      `(${num(run)} of ${num(attempted)}). The rest were displayed with the previous ` +
      `boxes, so plates in them were never read.`,
    )
  }
  if (cam.fps > 0 && cam.fps < 5 && frames > 60) {
    verdict = 'degraded'
    reasons.push(`Arriving at ${cam.fps.toFixed(1)} fps. Below about 5 fps a vehicle can cross ` +
                 `the frame between samples and never be read.`)
  }
  if ((cam.seconds_since_frame ?? 0) > 5) {
    verdict = 'down'
    reasons.push(`No frame for ${num(cam.seconds_since_frame ?? 0, 1)} s while marked ` +
                 `"${cam.status}".`)
  }
  if (frames > 300 && (cam.plates_detected_count ?? 0) === 0) {
    if (verdict === 'healthy') verdict = 'degraded'
    reasons.push(`${num(frames)} frames processed and no plate published yet. Either no ` +
                 `vehicle has passed, or the plate stage is not working on this view.`)
  }

  if (verdict === 'healthy') {
    reasons.push('Frames arriving, detector keeping pace, no detection failures.')
  }
  return { verdict, reasons, detectionRate, plateYield }
}

export default function CameraHealth() {
  const [slots, setSlots] = useState<MultiCamSlot[]>([])
  const [cameras, setCameras] = useState<Camera[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [checked, setChecked] = useState<string>('—')

  // Only the live slot status is polled. /api/health and /api/cameras are
  // database round-trips whose answers do not change second to second; polling
  // them here would add latency to every other tab for no new information.
  const refresh = useCallback(() => {
    getMultiCamStatus()
      .then(r => {
        setSlots(r.cameras ?? [])
        setChecked(new Date().toLocaleTimeString())
      })
      .catch(() => setSlots([]))
  }, [])

  useEffect(() => {
    refresh()
    getHealth().then(setHealth).catch(() => setHealth(null))
    getCameras().then(setCameras).catch(() => setCameras([]))
    const t = setInterval(refresh, 3000)
    return () => clearInterval(t)
  }, [refresh])

  const assessed = slots.map(s => ({ cam: s, a: assess(s) }))
  const counts = assessed.reduce<Record<string, number>>((acc, { a }) => {
    acc[a.verdict] = (acc[a.verdict] ?? 0) + 1
    return acc
  }, {})
  const rt = health?.runtime

  return (
    <div className="health-view">
      <Panel
        title="Live feed health"
        hint={
          <span className="hint-text">
            {assessed.length} feed(s) · {counts.healthy ?? 0} healthy · {counts.degraded ?? 0} degraded
            · {counts.down ?? 0} down · {counts.idle ?? 0} idle · checked {checked}
          </span>
        }
      >
        <p className="health-note">
          A feed can stream perfectly and still read nothing. These verdicts come from
          measured counters, and the rule that produced each one is written beside it.
        </p>

        {assessed.length === 0 ? (
          <Empty title="No camera slots reported">
            The backend is not returning multi-camera status. Check that the API is running.
          </Empty>
        ) : (
          <div className="health-grid">
            {assessed.map(({ cam, a }) => (
              <section key={cam.camera_id} className={`health-card is-${a.verdict}`}>
                <header className="health-head">
                  <div>
                    <span className="mono health-id">{cam.camera_id}</span>
                    <span className="health-name">{cam.name}</span>
                  </div>
                  <span className={`badge ${VERDICT_KIND[a.verdict]}`}>
                    {VERDICT_LABEL[a.verdict]}
                  </span>
                </header>

                <dl className="health-metrics">
                  <div><dt>Rate</dt><dd className="mono">{cam.fps > 0 ? `${cam.fps.toFixed(1)} fps` : '—'}</dd></div>
                  <div><dt>Frames</dt><dd className="mono">{num(cam.frame_count)}</dd></div>
                  <div>
                    <dt>Detector kept up</dt>
                    <dd className="mono">
                      {a.detectionRate === null ? '—' : `${(a.detectionRate * 100).toFixed(0)}%`}
                    </dd>
                  </div>
                  <div><dt>Plates</dt><dd className="mono">{num(cam.plates_detected_count ?? 0)}</dd></div>
                  <div>
                    <dt>Plates / 1k frames</dt>
                    <dd className="mono">{a.plateYield === null ? '—' : num(a.plateYield, 1)}</dd>
                  </div>
                  <div>
                    <dt>Detection errors</dt>
                    <dd className="mono">{num(cam.detect_failures ?? 0)}</dd>
                  </div>
                  <div>
                    <dt>Last frame</dt>
                    <dd className="mono">
                      {cam.seconds_since_frame === null || cam.seconds_since_frame === undefined
                        ? '—' : `${num(cam.seconds_since_frame, 1)} s ago`}
                    </dd>
                  </div>
                  <div><dt>Source</dt><dd>{cam.source_label || cam.source_type || '—'}</dd></div>
                </dl>

                <ul className="health-reasons">
                  {a.reasons.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </section>
            ))}
          </div>
        )}
      </Panel>

      <Panel
        title="Processing host"
        hint={<span className="hint-text">Everything above shares one GPU</span>}
      >
        <dl className="health-metrics wide">
          <div><dt>Device</dt><dd>{rt?.gpu_name ?? rt?.device ?? '—'}</dd></div>
          <div><dt>CUDA</dt><dd>{rt?.cuda ? 'available' : 'not available'}</dd></div>
          <div><dt>Feeds running</dt><dd className="mono">
            {slots.filter(s => s.status === 'playing').length} of {slots.length}
          </dd></div>
          <div><dt>Frames skipped for detection</dt><dd className="mono">
            {num(slots.reduce((n, s) => n + (s.detections_skipped ?? 0), 0))}
          </dd></div>
        </dl>
        <p className="health-note">
          Skipped frames are shown to the operator with the previous boxes drawn, never
          silently discarded. A number climbing quickly means this GPU is at its limit for
          the number of feeds running — add capacity or lower the phones' frame rate.
        </p>
      </Panel>

      <Panel
        title="Recorded run cameras"
        hint={<span className="hint-text">From the last batch pipeline run, not the live feeds</span>}
      >
        {cameras.length === 0 ? (
          <Empty title="No batch run on record">
            Process videos with the pipeline to populate this table.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Camera</th><th>Road</th>
                  <th className="ta-r">Frames</th><th className="ta-r">Rate</th>
                  <th className="ta-r">Plate reads</th><th className="ta-r">Per 1k frames</th>
                  <th>Location source</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map(c => (
                  <tr key={c.camera_id}>
                    <td className="mono">{c.camera_id}</td>
                    <td>{c.road || '—'}</td>
                    <td className="ta-r mono">{num(c.frames)}</td>
                    <td className="ta-r mono">{num(c.processing_fps, 1)} fps</td>
                    <td className="ta-r mono">{num(c.plate_dets)}</td>
                    <td className="ta-r mono">
                      {c.frames > 0 ? num((c.plate_dets / c.frames) * 1000, 1) : '—'}
                    </td>
                    <td>{c.location_source}</td>
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
