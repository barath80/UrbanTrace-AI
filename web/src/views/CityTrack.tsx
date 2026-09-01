/**
 * Query-based vehicle tracking across the whole camera network.
 *
 * Type a plate, get its journey: every sighting in chronological order with an
 * accurate timestamp and camera location, plotted on the GIS map, and the
 * measured leg between consecutive cameras with a verdict on whether that
 * journey was physically possible.
 *
 * Lookup tolerates OCR confusion — searching TN01AB1234 finds the sighting
 * that was read as TNO1AB1234 — but only where the difference is explainable
 * as a misread. Two genuinely different plates are never merged.
 */
import { useCallback, useEffect, useState } from 'react'

import {
  addNetWatchlist,
  getNetTrajectory,
  searchNetPlates,
  type Camera,
  type NetPlateHit,
  type NetTrajectory,
} from '../api'
import CityMap, { type RouteLine } from '../components/CityMap'
import { Empty, Loading, Panel, Tile, num, verdictKind } from '../components/ui'

const WINDOWS = [
  { label: '24 hours', hours: 24 },
  { label: '7 days', hours: 168 },
  { label: '30 days', hours: 720 },
]

export default function CityTrack({ initialPlate }: { initialPlate?: string } = {}) {
  const [query, setQuery] = useState(initialPlate ?? '')
  const [hours, setHours] = useState(168)
  const [hits, setHits] = useState<NetPlateHit[]>([])
  const [trajectory, setTrajectory] = useState<NetTrajectory | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const track = useCallback((plate: string, h: number) => {
    if (!plate.trim()) return
    setLoading(true)
    setError(null)
    setNote(null)
    getNetTrajectory(plate.trim().toUpperCase(), h)
      .then(setTrajectory)
      .catch(e => {
        setTrajectory(null)
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setLoading(false))
  }, [])

  const search = useCallback((q: string, h: number) => {
    if (!q.trim()) { setHits([]); return }
    searchNetPlates(q.trim().toUpperCase(), h).then(setHits).catch(() => setHits([]))
  }, [])

  useEffect(() => {
    if (initialPlate) { setQuery(initialPlate); track(initialPlate, hours) }
    // Only on first mount for a plate handed in from another view.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPlate])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    track(query, hours)
    search(query, hours)
  }

  /* --- map inputs ----------------------------------------------------- */

  const sightings = trajectory?.sightings ?? []
  const route: RouteLine | null = sightings.length > 1 ? {
    coordinates: sightings.map(s => [s.latitude, s.longitude] as [number, number]),
    // Straight lines between camera positions, not a surveyed road path.
    real: false,
    stops: sightings.map(s => ({
      lat: s.latitude, lon: s.longitude, label: s.camera_id,
      popup: `<b>${s.camera_id}</b> ${s.camera_name ?? ''}<br>` +
             `${new Date(s.observed_at).toLocaleString()}<br>` +
             `read as ${s.plate} at ${(s.confidence * 100).toFixed(0)}% confidence`,
    })),
  } : null

  const mapCameras: Camera[] = sightings.map(s => ({
    camera_id: s.camera_id, name: s.camera_name ?? s.camera_id, road: s.road,
    latitude: s.latitude, longitude: s.longitude, direction: s.direction ?? '—',
    video_file: '', capture_start: s.observed_at, location_source: 'config',
    has_video: false, frames: 0, processing_fps: 0, plate_dets: 1,
    vehicles_with_plate: 1, unique_vehicles: 1, vehicle_dets: 1,
    count_mode: null, count_traffic: 0, count_presence: 0,
    count_car: 0, count_motorcycle: 0, count_bus: 0, count_truck: 0,
  }))

  // Time spent TRAVELLING, not last-seen minus first-seen. The two differ
  // whenever the vehicle was out of sight long enough to break the journey,
  // and reporting the span as a duration turns two separate trips into one
  // impossibly slow one.
  const durationMin = trajectory?.travelling_minutes ?? null
  const spanMin = trajectory?.observed_span_minutes ?? null
  const breaks = trajectory?.breaks ?? []

  const speeds = (trajectory?.legs ?? [])
    .map(l => l.implied_speed_kmh).filter((v): v is number => v != null)
  const meanSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null
  const distance = (trajectory?.legs ?? [])
    .reduce((a, l) => a + (l.road_distance_km || 0), 0)

  return (
    <div className="net-view">
      <Panel title="Track a vehicle across the city">
        <form className="net-search" onSubmit={submit}>
          <input
            className="input plate"
            placeholder="Enter a licence plate — e.g. TN01AB1234"
            value={query}
            onChange={e => setQuery(e.target.value)}
            aria-label="Licence plate"
          />
          <select className="input sm" value={hours}
                  onChange={e => setHours(Number(e.target.value))}
                  aria-label="Search window">
            {WINDOWS.map(w => <option key={w.hours} value={w.hours}>{w.label}</option>)}
          </select>
          <button className="btn" type="submit" disabled={!query.trim() || loading}>
            Track
          </button>
          {trajectory && (
            <button
              className="btn ghost"
              type="button"
              onClick={() => addNetWatchlist(trajectory.plate, 'Added from the tracking view')
                .then(() => setNote(`${trajectory.plate} added to the watchlist.`))
                .catch(e => setError(String(e)))}
            >
              Add to watchlist
            </button>
          )}
        </form>

        {note && <div className="net-note">{note}</div>}
        {error && <div className="mc-error">{error}</div>}

        {hits.length > 1 && (
          <div className="net-hits">
            <span className="hint-text">Other spellings read in this window:</span>
            {hits.map(h => (
              <button key={h.plate} className="pill"
                      onClick={() => { setQuery(h.plate); track(h.plate, hours) }}>
                {h.plate} · {h.sightings} sighting(s) · {h.cameras} camera(s)
              </button>
            ))}
          </div>
        )}
      </Panel>

      {loading && <Loading what="Reconstructing trajectory…" />}

      {!loading && !trajectory && !error && (
        <Empty title="No vehicle selected">
          Enter a plate to reconstruct its journey across the camera network.
        </Empty>
      )}

      {trajectory && (
        <>
          <Panel title={`Journey — ${trajectory.plate}`}>
            <div className="tiles">
              <Tile label="Cameras" value={trajectory.cameras} sub="nodes that saw it" />
              <Tile label="Sightings" value={sightings.length} sub="confirmed reads" accent="var(--s1)" />
              <Tile label="Travelling time" value={durationMin} decimals={1}
                    sub="minutes, summed over legs" />
              <Tile label="Journeys" value={trajectory.journeys}
                    sub={breaks.length > 0 ? `${breaks.length} break(s) in between` : 'one continuous trip'}
                    accent={breaks.length > 0 ? 'var(--warning)' : undefined} />
              <Tile label="Distance" value={distance || null} decimals={2}
                    sub="km, straight-line estimate" />
              <Tile label="Mean speed" value={meanSpeed} decimals={1} sub="km/h between cameras" />
            </div>
            {spanMin != null && durationMin != null && spanMin > durationMin * 1.05 && (
              <p className="net-note">
                Seen over a {num(spanMin / 60, 1)} h window but only{' '}
                {num(durationMin, 1)} min of that was measured travel. The rest is
                time the vehicle spent out of sight between separate trips, and is
                deliberately not counted as journey time.
              </p>
            )}
            {trajectory.spellings.length > 1 && (
              <p className="net-note">
                Merged OCR spellings:{' '}
                <span className="mono">{trajectory.spellings.join(', ')}</span> — every
                difference is explainable as a character misread.
              </p>
            )}
          </Panel>

          <Panel title="Path on the city map"
                 hint={<span className="hint-text">
                   Markers are cameras that read the plate, in order. The line between
                   them is a straight-line estimate, not a surveyed road.
                 </span>}>
            {route ? (
              <div className="net-map">
                <CityMap cameras={mapCameras} route={route} showCameras pitch={0} />
              </div>
            ) : (
              <Empty title="Only one sighting">
                A path needs at least two cameras.
              </Empty>
            )}
          </Panel>

          <Panel title="Chronological history">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th><th>Camera</th><th>Location</th>
                    <th>Read as</th><th className="ta-r">Confidence</th>
                    <th className="ta-r">Frames</th><th>Format</th>
                  </tr>
                </thead>
                <tbody>
                  {sightings.map(s => (
                    <tr key={s.sighting_id}>
                      <td className="mono">{new Date(s.observed_at).toLocaleString()}</td>
                      <td className="mono">{s.camera_id}</td>
                      <td>{s.camera_name ?? '—'}</td>
                      <td className="mono">{s.plate}</td>
                      <td className="ta-r mono">{(s.confidence * 100).toFixed(0)}%</td>
                      <td className="ta-r mono">{s.support}/{s.total_readings}</td>
                      <td>
                        <span className={`badge ${s.is_valid_format ? 'ok' : 'warn'}`}>
                          {s.is_valid_format ? 'valid' : 'unrecognised'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {breaks.length > 0 && (
            <Panel title="Journey breaks"
                   hint={<span className="hint-text">
                     Gaps too long to be one continuous trip
                   </span>}>
              <ul className="mc-legs">
                {breaks.map((b, i) => (
                  <li key={i}>
                    <span className="mono">{b.after_camera} → {b.before_camera}</span>
                    <span className="mono">{num(b.gap_minutes / 60, 1)} h out of sight</span>
                    <span className="badge warn">SEPARATE JOURNEY</span>
                    <span className="mc-leg-note">{b.reason}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {trajectory.legs.length > 0 && (
            <Panel title="Movement between cameras"
                   hint={<span className="hint-text">
                     Each leg is checked against what a vehicle could physically do.
                   </span>}>
              <ul className="mc-legs">
                {trajectory.legs.map((l, i) => (
                  <li key={i}>
                    <span className="mono">{l.from_camera} → {l.to_camera}</span>
                    <span className="mono">
                      {num(l.travel_minutes, 1)} min · {num(l.road_distance_km, 2)} km ·{' '}
                      {l.implied_speed_kmh == null ? '—' : `${num(l.implied_speed_kmh, 0)} km/h`}
                    </span>
                    <span className={`badge ${verdictKind(l.verdict)}`}>{l.verdict}</span>
                    <span className="mc-leg-note">{l.reason}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </>
      )}
    </div>
  )
}
