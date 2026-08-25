/**
 * Watchlist alerts.
 *
 * Alerts fire ONLY for plates an operator put on the watchlist. An empty
 * watchlist means no alerts — the system never seeds its own list from what it
 * happened to see, because a list that populates itself is not a watchlist.
 */
import { useEffect, useState } from 'react'

import { api, getAlerts, type Alert } from '../api'
import { Badge, Empty, Loading, Panel, clock, num } from '../components/ui'

interface WatchRow { plate_number: string; note: string | null; active: boolean }

const sevKind = (s: string) =>
  s === 'HIGH' ? 'critical' : s === 'MEDIUM' ? 'warn' : 'muted'

export default function Alerts({ onTrack }: { onTrack: (plate: string) => void }) {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [watch, setWatch] = useState<WatchRow[]>([])
  const [busy, setBusy] = useState(true)
  const [add, setAdd] = useState('')

  const reload = () => {
    Promise.all([getAlerts(), api<WatchRow[]>('/watchlist')])
      .then(([a, w]) => { setAlerts(a); setWatch(w) })
      .catch(() => { /* empty states below */ })
      .finally(() => setBusy(false))
  }
  useEffect(reload, [])

  const addPlate = async () => {
    const p = add.trim().toUpperCase()
    if (!p) return
    await api(`/watchlist/${encodeURIComponent(p)}`, { method: 'POST' })
    setAdd('')
    reload()
  }

  if (busy) return <Panel><Loading /></Panel>

  return (
    <>
      <div className="note">
        Alerts fire only for plates on the watchlist. Adding a plate here affects
        the <b>next</b> pipeline run — alerts are generated during processing, against
        the reconstructed vehicle identity, so a plate read under a merged OCR variant
        still matches.
      </div>

      <div className="grid grid-2">
        <Panel title="Alerts" hint={`${alerts.length} in this run`}>
          {alerts.length === 0 ? (
            <Empty title="No alerts in this run">
              {watch.length === 0
                ? 'The watchlist is empty, so nothing could match.'
                : 'No watchlisted plate was seen by any camera in this run.'}
            </Empty>
          ) : alerts.map((a, i) => (
            <div className={`alert-card ${a.severity}`} key={i}>
              {a.evidence && (
                <img loading="lazy" alt=""
                     src={`/api/evidence?path=${encodeURIComponent(a.evidence)}`} />
              )}
              <div className="body">
                <div className="t">
                  <Badge kind={sevKind(a.severity)}>{a.severity}</Badge>
                  <Badge kind="info">{a.alert_type.replace(/_/g, ' ')}</Badge>
                  {a.plate_number && (
                    <span className="plate-cell" onClick={() => onTrack(a.plate_number!)}>
                      {a.plate_number}
                    </span>
                  )}
                  <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>
                    {a.camera_id} · {clock(a.occurred_at)} · conf {num(a.ocr_confidence, 3)}
                  </span>
                </div>
                <div className="r">{a.reason}</div>
              </div>
            </div>
          ))}
        </Panel>

        <Panel title="Watchlist" hint={`${watch.length} plate(s)`}>
          <div className="search-row">
            <input className="input plate sm" placeholder="Add a plate to the watchlist"
                   value={add} onChange={e => setAdd(e.target.value)}
                   onKeyDown={e => e.key === 'Enter' && addPlate()} />
            <button className="btn sm" onClick={addPlate} disabled={!add.trim()}>Add</button>
          </div>
          {watch.length === 0 ? (
            <Empty title="Watchlist is empty">
              No alerts can fire until a plate is added here or to{' '}
              <code className="mono">data/watchlist.txt</code>.
            </Empty>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Plate</th><th>Note</th><th>Active</th></tr>
                </thead>
                <tbody>
                  {watch.map(w => (
                    <tr key={w.plate_number}>
                      <td className="plate-cell" onClick={() => onTrack(w.plate_number)}>
                        {w.plate_number}
                      </td>
                      <td>{w.note ?? ''}</td>
                      <td>
                        <Badge kind={w.active ? 'ok' : 'muted'}>
                          {w.active ? 'active' : 'off'}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </>
  )
}
