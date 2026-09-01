/**
 * Query-based trajectory reconstruction: type a plate, get everything.
 *
 * One request to /api/vehicle-dossier assembles the sightings, the road path,
 * the per-leg verdicts, this vehicle's own heat surface, its alerts and its
 * journey statistics. Assembling them server-side is deliberate: when the
 * client stitched these together from separate endpoints, the halves disagreed
 * (the view once reported "2 cameras" and "1 sighting" for the same car).
 */
import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, getDossier, getVehicles, type Dossier, type VehicleRow } from '../api'
import CityMap, { type RouteLine } from '../components/CityMap'
import {
  Badge, Empty, Loading, Panel, Provenance, Tile, clock, hhmmss, num, verdictKind,
} from '../components/ui'

export default function TrackVehicle({ initialPlate }: { initialPlate?: string }) {
  const [query, setQuery] = useState(initialPlate ?? '')
  const [known, setKnown] = useState<VehicleRow[]>([])
  const [dossier, setDossier] = useState<Dossier | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { getVehicles().then(setKnown).catch(() => setKnown([])) }, [])

  const run = useCallback(async (plate: string) => {
    const p = plate.trim().toUpperCase()
    if (!p) return
    setBusy(true); setError(null)
    try {
      setDossier(await getDossier(p))
    } catch (e) {
      setDossier(null)
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => { if (initialPlate) run(initialPlate) }, [initialPlate, run])

  const route: RouteLine | null = useMemo(() => {
    if (!dossier) return null
    const path = dossier.path
    const coords: [number, number][] =
      path && path.coordinates.length > 1
        ? path.coordinates
        : dossier.timeline.map(t => [t.latitude, t.longitude])
    return {
      coordinates: coords,
      real: !!path?.is_real_road,
      stops: dossier.timeline.map((t, i) => ({
        lat: t.latitude, lon: t.longitude, label: String(i + 1),
        popup:
          `<b>${dossier.plate}</b> — stop ${i + 1} of ${dossier.timeline.length}<br>` +
          `${t.camera_id} · ${t.location ?? ''}<br>${clock(t.observed_at)}<br>` +
          `track #${t.local_track_id} · conf ${num(t.ocr_confidence, 3)}` +
          (t.is_merged_variant ? `<br>read here as ${t.plate_number}` : ''),
      })),
    }
  }, [dossier])

  return (
    <>
      <div className="note">
        Enter a plate to reconstruct its complete movement history across the camera
        network — sightings with timestamps and camera locations, the road it travelled,
        its own heat surface, per-leg plausibility verdicts and the raw OCR behind every
        identification.
      </div>

      <form
        className="search-row"
        onSubmit={e => { e.preventDefault(); run(query) }}
      >
        <input
          className="input plate"
          placeholder="Enter a licence plate — e.g. TN01AB1234"
          value={query}
          onChange={e => setQuery(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <button className="btn" type="submit" disabled={busy || !query.trim()}>
          {busy ? 'Tracking…' : 'Track vehicle'}
        </button>
      </form>

      {known.length > 0 && (
        <div className="pills" style={{ marginBottom: 16 }}>
          <span style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>Detected plates:</span>
          {known.slice(0, 14).map(v => (
            <button
              key={v.plate_number}
              className="pill"
              onClick={() => { setQuery(v.plate_number); run(v.plate_number) }}
            >
              {v.plate_number}
              {v.camera_count > 1 && ` · ${v.camera_count} cams`}
            </button>
          ))}
        </div>
      )}

      {busy && <Loading what="Reconstructing trajectory…" />}

      {error && !busy && (
        <Panel>
          <Empty title="No sightings found">{error}</Empty>
        </Panel>
      )}

      <AnimatePresence mode="wait">
        {dossier && !busy && (
          <motion.div
            key={dossier.plate}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <Dossierview d={dossier} route={route} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

function Dossierview({ d, route }: { d: Dossier; route: RouteLine | null }) {
  const s = d.stats
  const evidence = d.timeline.filter(t => t.evidence_crop)

  return (
    <>
      <div className="tiles">
        <div className="tile" style={{ ['--accent' as string]: 'var(--s1)' }}>
          <div className="label">Plate</div>
          <div className="value mono" style={{ fontSize: 22, color: 'var(--s1)' }}>{d.plate}</div>
          <div className="sub">
            {String((d.vehicle?.route_status as string) ?? '').replace(/_/g, ' ') || '—'}
          </div>
        </div>
        <Tile label="Cameras" value={s.cameras} sub={`${s.sightings} sightings`} />
        <Tile
          label="Road distance" value={s.road_distance_km} decimals={2}
          sub={<Provenance real={!!d.path?.is_real_road} />} accent="var(--s3)"
        />
        <Tile
          label="Journey time" value={s.measured_duration_min} decimals={1}
          sub={s.typical_duration_min != null
            ? `min measured · ${num(s.typical_duration_min, 1)} typical`
            : 'min measured'}
        />
        <Tile
          label="Mean speed" value={s.mean_implied_speed_kmh} decimals={1}
          sub="km/h implied over the road" accent="var(--s4)"
        />
        <Tile
          label="Alerts" value={d.alerts.length}
          sub={d.alerts.length ? 'watchlist match' : 'none'}
          accent={d.alerts.length ? 'var(--critical)' : undefined}
        />
      </div>

      <Panel
        title="Route on real roads, with this vehicle's heat surface"
        hint={<>
          <Provenance real={!!d.path?.is_real_road} />
          {d.path ? `${num(d.path.road_distance_km, 2)} km` : 'single camera'}
        </>}
        flush
      >
        <CityMap
          cameras={[]}
          heat={d.heat}
          heatScale="unit"
          route={route}
          showHeat
          showFlows={false}
          showCameras={false}
          className="map-canvas"
        />
        <div className="legend">
          <span className="legend-item">
            <span className="swatch line" style={{ background: 'var(--s2)' }} />
            Travelled route {d.path?.is_real_road ? '(real road)' : '(estimated, dashed)'}
          </span>
          <span className="legend-item">
            <span className="swatch" style={{ background: 'var(--critical)' }} />
            Sighting (measured)
          </span>
          <span className="legend-item">
            <span className="swatch" style={{ background: 'var(--s3)' }} />
            Inferred corridor
          </span>
        </div>
      </Panel>

      <div className="grid grid-2">
        <Panel title="Movement timeline">
          <div className="tl">
            {d.timeline.map((t, i) => {
              const leg = d.legs[i - 1]
              return (
                <div className="tl-item" key={t.observation_id}>
                  {leg && (
                    <div className="tl-leg">
                      <div className="head">
                        ↓ {num(leg.travel_minutes, 1)} min · {num(leg.road_distance_km, 2)} km ·{' '}
                        {num(leg.implied_speed_kmh, 0)} km/h
                        <Badge kind={verdictKind(leg.verdict)}>{leg.verdict}</Badge>
                        <Provenance real={leg.distance_source === 'graphhopper'} />
                      </div>
                      <div className="why">{leg.reason}</div>
                    </div>
                  )}
                  <div className="tl-cam">
                    {t.camera_id} — {t.location ?? ''}
                    {t.is_merged_variant && (
                      <Badge kind="warn">read as {t.plate_number}</Badge>
                    )}
                  </div>
                  <div className="tl-meta">
                    {clock(t.observed_at)} · video t={num(t.video_time_s, 2)}s · track #
                    {t.local_track_id} · conf {num(t.ocr_confidence, 3)}
                  </div>
                  <div className="tl-meta">
                    {num(t.latitude, 4)}, {num(t.longitude, 4)} · {t.vehicle_type ?? ''}
                  </div>
                </div>
              )
            })}
          </div>
        </Panel>

        <div>
          <Panel title="Journey analysis">
            <div className="fact"><span>Sightings</span><span>{s.sightings}</span></div>
            <div className="fact"><span>Cameras</span><span>{s.cameras}</span></div>
            <div className="fact">
              <span>Road distance</span><span>{num(s.road_distance_km, 2)} km</span>
            </div>
            <div className="fact">
              <span>Measured travel</span><span>{num(s.measured_duration_min, 1)} min</span>
            </div>
            <div className="fact">
              <span>Typical for road</span><span>{num(s.typical_duration_min, 1)} min</span>
            </div>
            <div className="fact">
              <span>Vs typical</span>
              <span style={{
                color: s.delay_vs_typical_min == null ? undefined
                  : s.delay_vs_typical_min > 2 ? 'var(--serious)'
                  : s.delay_vs_typical_min < -2 ? 'var(--warning)' : 'var(--ok)',
              }}>
                {s.delay_vs_typical_min == null ? '—'
                  : `${s.delay_vs_typical_min > 0 ? '+' : ''}${num(s.delay_vs_typical_min, 1)} min`}
              </span>
            </div>
            <div className="fact">
              <span>Mean implied speed</span><span>{num(s.mean_implied_speed_kmh, 1)} km/h</span>
            </div>
            <div className="fact">
              <span>Leg verdicts</span>
              <span>
                {Object.entries(s.verdicts).length
                  ? Object.entries(s.verdicts).map(([k, v]) => (
                      <Badge key={k} kind={verdictKind(k)}>{k} ×{v}</Badge>
                    ))
                  : '—'}
              </span>
            </div>
          </Panel>

          <Panel title="Where these numbers come from">
            {Object.entries(s.provenance).map(([k, v]) => (
              <div className="fact" key={k}>
                <span>{k.replace(/_/g, ' ')}</span>
                <span style={{ fontFamily: 'var(--sans)', color: 'var(--ink-2)' }}>{v}</span>
              </div>
            ))}
          </Panel>

          {d.alerts.length > 0 && (
            <Panel title="Alerts for this vehicle">
              {d.alerts.map((a, i) => (
                <div className={`alert-card ${a.severity}`} key={i}>
                  <div className="body">
                    <div className="t">
                      <Badge kind={a.severity === 'HIGH' ? 'critical'
                        : a.severity === 'MEDIUM' ? 'warn' : 'muted'}>{a.severity}</Badge>
                      <Badge kind="info">{a.alert_type.replace(/_/g, ' ')}</Badge>
                      <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>
                        {clock(a.occurred_at)}
                      </span>
                    </div>
                    <div className="r">{a.reason}</div>
                  </div>
                </div>
              ))}
            </Panel>
          )}
        </div>
      </div>

      {evidence.length > 0 && (
        <Panel title="Evidence" hint="plate crops from the source video">
          <div className="ev-grid">
            {evidence.map(t => (
              <div className="ev-card" key={t.observation_id}>
                <img
                  loading="lazy" alt={`${d.plate} at ${t.camera_id}`}
                  src={`/api/evidence?path=${encodeURIComponent(t.evidence_crop!)}`}
                />
                <div className="cap">
                  <div className="p">{t.plate_number}</div>
                  <div className="m">{t.camera_id} · {hhmmss(t.observed_at)}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel
        title="Per-frame OCR audit trail"
        hint={`${d.raw_readings.length} raw readings behind this identity`}
        flush
      >
        <div className="table-wrap" style={{ maxHeight: 420, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Observation</th><th className="num">Frame</th><th>Raw OCR</th>
                <th>After validation</th><th className="num">OCR conf</th>
                <th className="num">Plate conf</th><th>Variant</th><th>Repaired</th>
              </tr>
            </thead>
            <tbody>
              {d.raw_readings.map((r, i) => (
                <tr key={`${r.observation_id}-${r.frame_number}-${i}`}>
                  <td className="mono" style={{ fontSize: 11 }}>{r.observation_id}</td>
                  <td className="num">{r.frame_number}</td>
                  <td className="mono" style={{
                    color: r.text_raw !== r.text_final ? 'var(--warning)' : undefined,
                  }}>{r.text_raw}</td>
                  <td className="mono">{r.text_final}</td>
                  <td className="num">{num(r.ocr_confidence, 3)}</td>
                  <td className="num">{num(r.plate_confidence, 3)}</td>
                  <td>{r.variant}</td>
                  <td>{r.repaired ? <Badge kind="warn">repaired</Badge> : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  )
}
