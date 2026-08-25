/** The raw record: every timestamped plate observation the pipeline produced. */
import { useEffect, useMemo, useState } from 'react'

import { getObservations } from '../api'
import { Badge, Empty, Loading, Panel, clock, num } from '../components/ui'

type Row = Record<string, unknown>

export default function Observations({ onTrack }: { onTrack: (plate: string) => void }) {
  const [rows, setRows] = useState<Row[]>([])
  const [busy, setBusy] = useState(true)
  const [q, setQ] = useState('')

  useEffect(() => {
    getObservations(1000).then(setRows).catch(() => setRows([])).finally(() => setBusy(false))
  }, [])

  const shown = useMemo(() => {
    const s = q.trim().toUpperCase()
    if (!s) return rows
    return rows.filter(r =>
      String(r.plate_number ?? '').toUpperCase().includes(s) ||
      String(r.camera_id ?? '').toUpperCase().includes(s))
  }, [rows, q])

  if (busy) return <Panel><Loading /></Panel>

  return (
    <Panel
      title="Observations"
      hint={`${shown.length} of ${rows.length} — one row per vehicle, per camera`}
      flush
    >
      <div style={{ padding: '13px 16px', borderBottom: '1px solid var(--border-soft)' }}>
        <input className="input sm" placeholder="Filter by plate or camera…"
               value={q} onChange={e => setQ(e.target.value)} />
      </div>
      {shown.length === 0 ? (
        <Empty title="No observations">
          {rows.length ? 'Nothing matches that filter.' : 'This run produced no plate reads.'}
        </Empty>
      ) : (
        <div className="table-wrap" style={{ maxHeight: '68vh', overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Plate</th><th>Camera</th><th>Observed at</th>
                <th className="num">Video t</th><th className="num">Frame</th>
                <th>Type</th><th className="num">OCR conf</th>
                <th className="num">Support</th><th className="num">Agreement</th>
                <th>Format</th><th>Method</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r, i) => (
                <tr key={String(r.observation_id ?? i)}>
                  <td className="plate-cell"
                      onClick={() => onTrack(String(r.plate_number))}>
                    {String(r.plate_number)}
                  </td>
                  <td className="mono">{String(r.camera_id)}</td>
                  <td className="mono" style={{ fontSize: 11.5 }}>
                    {clock(r.observed_at as string)}
                  </td>
                  <td className="num">{num(r.video_time_s, 2)}s</td>
                  <td className="num">{num(r.frame_number)}</td>
                  <td>{String(r.vehicle_type ?? '')}</td>
                  <td className="num">{num(r.ocr_confidence, 3)}</td>
                  <td className="num">{num(r.support)}/{num(r.total_readings)}</td>
                  <td className="num">{num(r.agreement, 2)}</td>
                  <td>
                    <Badge kind={r.is_valid_format ? 'ok' : 'warn'}>
                      {r.is_valid_format ? 'valid' : 'non-standard'}
                    </Badge>
                  </td>
                  <td style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>
                    {String(r.aggregation_method ?? '')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}
