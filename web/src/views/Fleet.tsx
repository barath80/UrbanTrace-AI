/**
 * Fleet: the camera estate, the ingest workers and the health of every feed.
 *
 * All of it is read from the database the workers report into. A feed whose
 * worker has stopped reporting shows as "unmonitored" instead of keeping its
 * last known state, so a dead worker cannot make cameras look healthy.
 *
 * Cameras are added here (one, or many from a CSV) and picked up by the
 * workers within a refresh interval; no restart is involved.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  deleteNetCamera,
  getApiKey,
  getFeeds,
  getFleet,
  getWorkers,
  saveNetCamera,
  saveNetCamerasBulk,
  setApiKey,
  setNetCameraEnabled,
  type NetCameraInput,
  type NetFeed,
  type NetFleet,
  type NetWorker,
} from '../api'
import { Badge, Empty, Loading, Panel, Tile, clock, num } from '../components/ui'

const STATE_KIND: Record<string, string> = {
  live: 'ok',
  connecting: 'info',
  reconnecting: 'warn',
  stopped: 'muted',
  unmonitored: 'serious',
  disabled: 'muted',
  'push-only': 'muted',
}

const BLANK = {
  camera_id: '', name: '', latitude: '', longitude: '',
  road: '', sector: '', direction: '', source: '',
}

/** CSV with a header row: camera_id,name,latitude,longitude[,road,sector,direction,source]. */
function parseCsv(text: string): NetCameraInput[] {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean)
  if (lines.length < 2) throw new Error('The CSV needs a header row and at least one camera.')
  const head = lines[0].split(',').map(h => h.trim().toLowerCase())
  for (const need of ['camera_id', 'name', 'latitude', 'longitude']) {
    if (!head.includes(need)) throw new Error(`The CSV header is missing "${need}".`)
  }
  return lines.slice(1).map((line, i) => {
    const cells = line.split(',').map(c => c.trim())
    const row: Record<string, string> = {}
    head.forEach((h, k) => { row[h] = cells[k] ?? '' })
    const latitude = Number(row.latitude)
    const longitude = Number(row.longitude)
    if (!row.camera_id || Number.isNaN(latitude) || Number.isNaN(longitude)) {
      throw new Error(`Row ${i + 2}: camera_id, latitude and longitude must be valid.`)
    }
    return {
      camera_id: row.camera_id, name: row.name || row.camera_id, latitude, longitude,
      road: row.road, sector: row.sector, direction: row.direction || 'UNKNOWN',
      source: row.source,
    }
  })
}

export default function Fleet() {
  const [fleet, setFleet] = useState<NetFleet | null>(null)
  const [workers, setWorkers] = useState<NetWorker[]>([])
  const [feeds, setFeeds] = useState<NetFeed[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('all')
  const [form, setForm] = useState(BLANK)
  const [key, setKey] = useState(getApiKey())
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    Promise.all([getFleet(), getWorkers(), getFeeds()])
      .then(([f, w, fe]) => { setFleet(f); setWorkers(w); setFeeds(fe); setError(null) })
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [load])

  const act = async (fn: () => Promise<unknown>, done: string) => {
    setSaving(true)
    setNotice(null)
    try {
      await fn()
      setNotice(done)
      load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const add = () => {
    const latitude = Number(form.latitude)
    const longitude = Number(form.longitude)
    if (!form.camera_id || !form.name || form.latitude === '' || form.longitude === ''
      || Number.isNaN(latitude) || Number.isNaN(longitude)) {
      setError('Camera id, name, latitude and longitude are required.')
      return
    }
    void act(async () => {
      await saveNetCamera({
        camera_id: form.camera_id.trim(), name: form.name.trim(), latitude, longitude,
        road: form.road, sector: form.sector, direction: form.direction || 'UNKNOWN',
        source: form.source.trim(),
      })
      setForm(BLANK)
    }, `Saved ${form.camera_id}. Workers pick it up within about 15 seconds.`)
  }

  const importFile = (file: File) => {
    void file.text()
      .then(text => parseCsv(text))
      .then(cams => act(() => saveNetCamerasBulk(cams), `Imported ${cams.length} camera(s).`))
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
  }

  const shown = useMemo(() => feeds.filter(f =>
    (stateFilter === 'all' || f.state === stateFilter)
    && (!filter || `${f.camera_id} ${f.name} ${f.sector ?? ''}`.toLowerCase()
      .includes(filter.toLowerCase()))), [feeds, filter, stateFilter])

  const states = useMemo(() => Array.from(new Set(feeds.map(f => f.state))).sort(), [feeds])

  if (loading) return <Panel><Loading /></Panel>

  return (
    <>
      {error && <div className="note">{error}</div>}
      {notice && <div className="note">{notice}</div>}

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Tile label="Cameras" value={fleet?.cameras ?? 0}
          sub={`${fleet?.feeds_by_state.live ?? 0} live`} />
        <Tile label="Workers alive" value={fleet?.workers_alive ?? 0}
          sub={`${fleet?.workers ?? 0} registered`} />
        <Tile label="Needing attention"
          value={(fleet?.feeds_by_state.reconnecting ?? 0) + (fleet?.feeds_by_state.unmonitored ?? 0)}
          sub="reconnecting or unmonitored" />
      </div>

      <Panel title="Ingest workers"
        hint="Cameras are spread across live workers by location; a lost worker's cameras move within about 20 seconds">
        {workers.length === 0 ? (
          <Empty title="No workers have registered">
            Start one with <code className="mono">python -m urbantrace.scale.worker</code>.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr>
                <th>Worker</th><th>Host</th><th>Status</th><th className="num">Cameras</th>
                <th className="num">Frames/s</th><th className="num">Per-camera fps</th>
                <th>Last heartbeat</th>
              </tr></thead>
              <tbody>
                {workers.map(w => (
                  <tr key={w.worker_id}>
                    <td className="mono">{w.worker_id}</td>
                    <td>{w.host ?? '—'}</td>
                    <td><Badge kind={w.alive ? 'ok' : 'critical'}>{w.alive ? 'alive' : 'silent'}</Badge></td>
                    <td className="num">{num(w.stats.cameras_owned)}</td>
                    <td className="num">{num(w.stats.throughput_fps, 1)}</td>
                    <td className="num">
                      {w.stats.rate
                        ? `${num(w.stats.rate.current_fps, 1)} / ${num(w.stats.rate.target_fps, 1)}`
                        : '—'}
                    </td>
                    <td>{w.age_s === null ? '—' : `${num(w.age_s, 0)} s ago`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Feeds" hint={`${shown.length} of ${feeds.length}`} flush>
        <div className="search-row" style={{ padding: 12 }}>
          <input className="input sm" placeholder="Filter by id, name or sector"
            value={filter} onChange={e => setFilter(e.target.value)} />
          <select className="input sm" value={stateFilter}
            onChange={e => setStateFilter(e.target.value)}>
            <option value="all">All states</option>
            {states.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        {feeds.length === 0 ? (
          <Empty title="No cameras yet">Add one below, or import a CSV.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr>
                <th>Camera</th><th>Sector</th><th>State</th><th>Worker</th>
                <th className="num">Source fps</th><th>Resolution</th>
                <th className="num">Analysed</th><th className="num">Static skipped</th>
                <th className="num">Reconnects</th><th>Last frame</th><th>Last error</th><th />
              </tr></thead>
              <tbody>
                {shown.map(f => (
                  <tr key={f.camera_id}>
                    <td><b className="mono">{f.camera_id}</b> {f.name}</td>
                    <td>{f.sector || '—'}</td>
                    <td><Badge kind={STATE_KIND[f.state] ?? 'muted'}>{f.state}</Badge></td>
                    <td className="mono">{f.worker_id ?? '—'}</td>
                    <td className="num">{num(f.source_fps, 1)}</td>
                    <td>{f.resolution ?? '—'}</td>
                    <td className="num">{num(f.frames_submitted)}</td>
                    <td className="num">{num(f.frames_skipped_static)}</td>
                    <td className="num">{num(f.reconnects)}</td>
                    <td>{clock(f.last_frame_at)}</td>
                    <td>{f.last_error ?? '—'}</td>
                    <td>
                      <button className="btn ghost sm" disabled={saving}
                        onClick={() => void act(
                          () => setNetCameraEnabled(f.camera_id, !f.enabled),
                          `${f.camera_id} ${f.enabled ? 'disabled' : 'enabled'}.`)}>
                        {f.enabled ? 'Disable' : 'Enable'}
                      </button>{' '}
                      <button className="btn ghost sm" disabled={saving}
                        onClick={() => {
                          if (window.confirm(`Delete ${f.camera_id}? Its sightings are kept.`)) {
                            void act(() => deleteNetCamera(f.camera_id), `Deleted ${f.camera_id}.`)
                          }
                        }}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Add a camera" hint="Stream credentials stay on the server and are redacted in every response">
        <div className="search-row" style={{ flexWrap: 'wrap', gap: 10 }}>
          {([
            ['camera_id', 'Camera id'], ['name', 'Name'], ['latitude', 'Latitude'],
            ['longitude', 'Longitude'], ['road', 'Road'], ['sector', 'Sector'],
            ['direction', 'Direction'],
          ] as const).map(([k, label]) => (
            <label className="field" key={k}>
              <span>{label}</span>
              <input className="input sm" value={form[k]}
                onChange={e => setForm({ ...form, [k]: e.target.value })} />
            </label>
          ))}
          <label className="field grow">
            <span>Stream URL (rtsp://, http://, file path)</span>
            <input className="input sm" value={form.source}
              onChange={e => setForm({ ...form, source: e.target.value })} />
          </label>
          <button className="btn" disabled={saving} onClick={add}>Save camera</button>
        </div>
        <div className="search-row" style={{ marginTop: 14 }}>
          <label className="field">
            <span>Import CSV (camera_id,name,latitude,longitude,road,sector,direction,source)</span>
            <input type="file" accept=".csv,text/csv"
              onChange={e => {
                const f = e.target.files?.[0]
                if (f) importFile(f)
                e.target.value = ''
              }} />
          </label>
        </div>
      </Panel>

      <Panel title="API key" hint="Only needed when the server sets UT_API_KEY">
        <div className="search-row">
          <input className="input sm" type="password" placeholder="API key" value={key}
            onChange={e => setKey(e.target.value)} />
          <button className="btn ghost sm"
            onClick={() => { setApiKey(key); setNotice('API key saved in this browser.') }}>
            Save
          </button>
        </div>
      </Panel>
    </>
  )
}
