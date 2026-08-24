/* UrbanTrace AI dashboard.
   Every value rendered here comes from the API, which reads Supabase.
   Nothing is generated client-side; empty data renders an explicit empty state. */

const S = { runId: null, cameras: [], vehicles: [], map: null, layers: {}, tmap: null };

const C = {
  s1: '#3987e5', s2: '#d95926', s3: '#199e70',
  good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b',
  muted: '#898781'
};

const api = async (path) => {
  const sep = path.includes('?') ? '&' : '?';
  const url = S.runId ? `/api${path}${sep}run_id=${S.runId}` : `/api${path}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

const peakSafe = (hm) => Math.max(...hm.map(h => h.observations), 1);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = (v, d = 0) => (v === null || v === undefined || v === '') ? '—' : Number(v).toFixed(d);
const clock = (ts) => ts ? String(ts).replace('T', ' ').replace(/\+.*$/, '').slice(0, 19) : '—';
const hhmmss = (ts) => ts ? clock(ts).slice(11) : '—';

const empty = (title, msg) =>
  `<div class="empty"><h3>${esc(title)}</h3><p>${msg}</p></div>`;
const NO_RUN = empty('No pipeline run yet',
  'Add videos to <code>data/videos/</code> and run <code>UrbanTrace_AI_Realtime.ipynb</code>. ' +
  'Results are written to Supabase and appear here automatically.');

const statusBadge = (s) => {
  const k = { COMPLETE_CORRIDOR: 'complete', PARTIAL_ROUTE: 'partial',
              DROPPED_AT_ORIGIN: 'dropped', SINGLE_CAMERA_NETWORK: 'single' }[s] || 'single';
  return `<span class="badge ${k}">${esc((s || '').replace(/_/g, ' '))}</span>`;
};
const verdictBadge = (v) =>
  `<span class="badge ${(v || '').toLowerCase()}">${esc(v)}</span>`;

/* ------------------------------------------------------------ tabs */
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('view-' + t.dataset.view).classList.add('active');
  if (t.dataset.view === 'map') {
    // Leaflet + leaflet.heat need a laid-out container: a heat layer built on a
    // display:none element throws "source width is 0". Build it on first reveal.
    if (!S.mapLoaded) { S.mapLoaded = true; loadMap().catch(e => {
      document.getElementById('map').innerHTML = empty('Map unavailable', esc(e.message)); }); }
    else if (S.map) setTimeout(() => S.map.invalidateSize(), 60);
  }
});

/* ---------------------------------------------------------- health */
async function loadHealth() {
  try {
    const h = await fetch('/api/health').then(r => r.json());
    const db = h.database || {};
    document.getElementById('dot-db').className = 'dot ' + (db.connected ? 'ok' : 'bad');
    document.getElementById('txt-db').textContent = db.connected
      ? `Supabase · ${db.public_tables} tables` : 'database offline';
    const rt = h.runtime || {};
    document.getElementById('dot-gpu').className = 'dot ' + (rt.cuda ? 'ok' : 'warn');
    document.getElementById('txt-gpu').textContent = rt.cuda
      ? (rt.gpu_name || 'GPU').replace('NVIDIA GeForce ', '') : 'CPU';
  } catch {
    document.getElementById('dot-db').className = 'dot bad';
    document.getElementById('txt-db').textContent = 'API unreachable';
  }
}

async function loadRuns() {
  const sel = document.getElementById('run-select');
  try {
    const runs = await fetch('/api/runs').then(r => r.json());
    if (!runs.length) { sel.innerHTML = '<option>no runs</option>'; return false; }
    sel.innerHTML = runs.map(r =>
      `<option value="${r.run_id}">${clock(r.started_at)} · ${r.camera_count} cam · ${r.observation_count ?? 0} obs</option>`
    ).join('');
    S.runId = runs[0].run_id;
    sel.onchange = () => { S.runId = sel.value; loadAll(); };
    return true;
  } catch { sel.innerHTML = '<option>unavailable</option>'; return false; }
}

/* -------------------------------------------------------- overview */
async function loadOverview() {
  const s = await api('/summary');
  document.getElementById('tiles').innerHTML = [
    ['Cameras', s.cameras ?? 0, `${s.video_count ?? 0} video sources`, ''],
    ['Vehicles identified', s.unique_plates ?? 0, `${s.observations ?? 0} observations`, 'accent'],
    ['Cross-camera routes', s.multi_camera ?? 0, 'tracked across cameras', ''],
    ['Avg speed', s.avg_speed_kmh ? num(s.avg_speed_kmh, 1) : '—',
     s.avg_speed_kmh ? 'km/h · estimated' : 'needs 2+ cameras', ''],
    ['Avg travel time', s.avg_travel_min ? num(s.avg_travel_min, 1) : '—',
     s.avg_travel_min ? 'minutes between cameras' : '—', ''],
    ['Active alerts', s.alerts ?? 0, 'watchlist + anomalies', (s.alerts > 0 ? 'alert' : '')],
    ['Mean OCR confidence', s.mean_ocr_conf ? num(s.mean_ocr_conf, 3) : '—',
     `${s.raw_readings ?? 0} raw frame reads`, ''],
    ['Compute', (s.device || '—').toUpperCase(), s.gpu_name ? String(s.gpu_name).replace('NVIDIA GeForce ', '') : '', '']
  ].map(([l, v, sub, cls]) =>
    `<div class="tile"><div class="label">${l}</div>
     <div class="value ${cls}">${esc(v)}</div><div class="sub">${esc(sub)}</div></div>`).join('');

  // per-camera vehicle counts
  const counts = await api('/camera-counts');
  const cbox = document.getElementById('chart-cameras');
  if (!counts.length) cbox.innerHTML = empty('No detections', 'No plates were read in this run.');
  else {
    const max = Math.max(...counts.map(c => c.vehicles_with_plate), 1);
    cbox.innerHTML = counts.map(c => `
      <div class="bar-row">
        <span class="name" title="${esc(c.location || c.camera_id)}">${esc(c.camera_id)} · ${esc(c.location || '')}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${c.vehicles_with_plate / max * 100}%;background:${C.s1}"></span></span>
        <span class="val">${c.vehicles_with_plate}</span>
      </div>`).join('');
  }

  // route outcomes
  S.vehicles = await api('/vehicles');
  const tally = {};
  S.vehicles.forEach(v => tally[v.route_status] = (tally[v.route_status] || 0) + 1);
  const colours = { COMPLETE_CORRIDOR: C.good, PARTIAL_ROUTE: C.warning,
                    DROPPED_AT_ORIGIN: C.critical, SINGLE_CAMERA_NETWORK: C.muted };
  const sbox = document.getElementById('chart-status');
  const keys = Object.keys(tally);
  if (!keys.length) sbox.innerHTML = empty('No vehicles', 'Nothing to summarise yet.');
  else {
    const max = Math.max(...Object.values(tally), 1);
    sbox.innerHTML = keys.map(k => `
      <div class="bar-row">
        <span class="name">${esc(k.replace(/_/g, ' '))}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${tally[k] / max * 100}%;background:${colours[k] || C.muted}"></span></span>
        <span class="val">${tally[k]}</span>
      </div>`).join('');
  }

  // routes table
  const rt = document.getElementById('table-routes');
  if (!S.vehicles.length) rt.innerHTML = NO_RUN;
  else rt.innerHTML = `<table><thead><tr>
      <th>Plate</th><th>Route</th><th>Status</th><th class="num">Cameras</th>
      <th>First seen</th><th>Last seen</th><th class="num">Duration</th><th class="num">Confidence</th>
    </tr></thead><tbody>${S.vehicles.map(v => `
      <tr class="clickable" onclick="trackPlate('${esc(v.plate_number)}')">
        <td class="plate">${esc(v.plate_number)}</td>
        <td style="font-family:var(--mono);font-size:11.5px">${esc(v.route_display || '—')}</td>
        <td>${statusBadge(v.route_status)}</td>
        <td class="num">${v.camera_count}</td>
        <td class="time">${clock(v.first_seen)}</td>
        <td class="time">${clock(v.last_seen)}</td>
        <td class="num">${v.total_duration_minutes ? num(v.total_duration_minutes, 1) + ' min' : '—'}</td>
        <td class="num">${num(v.confidence, 3)}</td>
      </tr>`).join('')}</tbody></table>`;
}

/* -------------------------------------------------------------- map */
async function loadMap() {
  S.cameras = await api('/cameras');
  const hm = await api('/heatmap');
  const flows = await api('/flows');
  document.getElementById('map-hint').textContent =
    `${S.cameras.length} cameras · ${flows.length} active flows`;
  if (!S.cameras.length) { document.getElementById('map').innerHTML = NO_RUN; return; }

  if (!S.map) {
    S.map = L.map('map', { zoomControl: true });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      { attribution: '© OpenStreetMap © CARTO', maxZoom: 19 }).addTo(S.map);
  }
  Object.values(S.layers).forEach(l => S.map.removeLayer(l));
  S.layers = {};

  // leaflet.heat draws onto a canvas sized from the container. If it is added
  // (or redrawn) while the container still has zero width it throws
  // "getImageData ... source width is 0". Size the map first, then add heat on
  // the next frame so the canvas has real dimensions.
  const heatPoints = hm.filter(h => h.observations > 0)
                       .map(h => [h.latitude, h.longitude, h.observations / peakSafe(hm)]);
  S.map.invalidateSize();
  requestAnimationFrame(() => {
    const box = document.getElementById('map');
    if (!box || box.clientWidth === 0 || !heatPoints.length) return;
    try {
      S.layers.heat = L.heatLayer(heatPoints, {
        radius: 46, blur: 34, minOpacity: .32,
        gradient: { 0.2: C.s3, 0.5: C.warning, 0.8: C.serious, 1: C.critical }
      }).addTo(S.map);
    } catch (e) { console.warn('heat layer skipped:', e.message); }
  });

  const maxFlow = Math.max(...flows.map(f => f.vehicles), 1);
  const byId = Object.fromEntries(S.cameras.map(c => [c.camera_id, c]));
  const flowLayer = L.layerGroup().addTo(S.map);
  flows.forEach(f => {
    const a = byId[f.from_camera], b = byId[f.to_camera];
    if (!a || !b) return;
    L.polyline([[a.latitude, a.longitude], [b.latitude, b.longitude]], {
      color: C.s1, weight: 2 + 8 * (f.vehicles / maxFlow), opacity: .8
    }).bindPopup(
      `<b>${esc(f.from_camera)} → ${esc(f.to_camera)}</b><br>${f.vehicles} vehicle(s)<br>` +
      `${num(f.road_distance_km, 2)} km · ${num(f.mean_travel_minutes, 1)} min · ` +
      `${num(f.mean_speed_kmh, 1)} km/h`).addTo(flowLayer);
  });
  S.layers.flows = flowLayer;

  const camLayer = L.layerGroup().addTo(S.map);
  S.cameras.forEach(c => {
    L.circleMarker([c.latitude, c.longitude], {
      radius: 9, color: '#fff', weight: 2, fillColor: C.s1, fillOpacity: .95
    }).bindPopup(
      `<b>${esc(c.camera_id)}</b> — ${esc(c.name)}<br>${esc(c.road || '')}<br>` +
      `${num(c.latitude, 4)}, ${num(c.longitude, 4)}<br>` +
      `<hr style="border:0;border-top:1px solid #333;margin:6px 0">` +
      `${c.vehicles_with_plate} vehicles identified<br>` +
      `${c.vehicle_dets} vehicle detections · ${c.plate_dets} plate detections<br>` +
      `location source: ${esc(c.location_source)}`).addTo(camLayer);
  });
  S.layers.cams = camLayer;

  S.map.fitBounds(L.latLngBounds(S.cameras.map(c => [c.latitude, c.longitude])).pad(0.28));
  setTimeout(() => S.map.invalidateSize(), 80);
}

/* --------------------------------------------------------- tracking */
window.trackPlate = async function (plate) {
  document.querySelector('.tab[data-view="track"]').click();
  document.getElementById('plate-input').value = plate;
  const box = document.getElementById('track-result');
  box.innerHTML = '<div class="loading">reconstructing trajectory…</div>';
  let d;
  try { d = await api(`/vehicle/${encodeURIComponent(plate)}`); }
  catch (e) { box.innerHTML = empty('Not found', esc(e.message)); return; }

  const v = d.vehicle || {};
  box.innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="label">Plate</div><div class="value accent" style="font-family:var(--mono);font-size:22px">${esc(d.plate)}</div>
        <div class="sub">${v.route_status ? esc(v.route_status.replace(/_/g, ' ')) : ''}</div></div>
      <div class="tile"><div class="label">Cameras</div><div class="value">${v.camera_count ?? d.timeline.length}</div><div class="sub">${d.timeline.length} sightings</div></div>
      <div class="tile"><div class="label">First seen</div><div class="value" style="font-size:18px">${hhmmss(v.first_seen || d.timeline[0].observed_at)}</div><div class="sub">${clock(v.first_seen || d.timeline[0].observed_at).slice(0, 10)}</div></div>
      <div class="tile"><div class="label">Last seen</div><div class="value" style="font-size:18px">${hhmmss(v.last_seen || d.timeline.at(-1).observed_at)}</div><div class="sub">${v.total_duration_minutes ? num(v.total_duration_minutes, 1) + ' min journey' : ''}</div></div>
      <div class="tile"><div class="label">Confidence</div><div class="value">${num(v.confidence ?? d.timeline[0].ocr_confidence, 3)}</div><div class="sub">${d.raw_readings.length} raw frame reads</div></div>
    </div>
    <div class="grid-2">
      <div class="panel"><div class="panel-head"><h2>Movement timeline</h2></div>
        <div class="panel-body"><div class="timeline">${d.timeline.map((t, i) => {
          const leg = d.legs[i - 1];
          return `<div class="tl-item">
            ${leg ? `<div class="tl-leg">↓ ${num(leg.travel_minutes, 1)} min · ${num(leg.road_distance_km, 2)} km ·
              ${num(leg.implied_speed_kmh, 0)} km/h ${verdictBadge(leg.verdict)}<br>
              <span style="color:var(--ink-muted)">${esc(leg.reason)}</span></div>` : ''}
            <div class="tl-cam">${esc(t.camera_id)} — ${esc(t.location || '')}</div>
            <div class="tl-meta">${clock(t.observed_at)} · video t=${num(t.video_time_s, 2)}s ·
              track #${t.local_track_id} · conf ${num(t.ocr_confidence, 3)}</div>
            <div class="tl-meta">${num(t.latitude, 4)}, ${num(t.longitude, 4)} · ${esc(t.vehicle_type || '')}</div>
          </div>`; }).join('')}</div></div></div>
      <div class="panel"><div class="panel-head"><h2>Route on map</h2></div>
        <div class="panel-body flush"><div id="trajectory-map"></div></div></div>
    </div>
    ${d.timeline.some(t => t.evidence_crop) ? `<div class="panel">
      <div class="panel-head"><h2>Evidence</h2><span class="hint">plate crops from source video</span></div>
      <div class="panel-body"><div class="evidence-grid">${d.timeline.filter(t => t.evidence_crop).map(t => `
        <div class="evidence-card"><img loading="lazy" src="/api/evidence?path=${encodeURIComponent(t.evidence_crop)}" alt="">
        <div class="cap"><div class="p">${esc(d.plate)}</div><div class="m">${esc(t.camera_id)} · ${hhmmss(t.observed_at)}</div></div></div>`).join('')}
      </div></div></div>` : ''}
    <div class="panel"><div class="panel-head"><h2>Per-frame OCR audit trail</h2>
      <span class="hint">every raw reading behind this identity</span></div>
      <div class="panel-body flush table-wrap"><table><thead><tr>
        <th>Observation</th><th class="num">Frame</th><th>Raw OCR</th><th>After validation</th>
        <th class="num">OCR conf</th><th class="num">Plate conf</th><th>Variant</th><th>Repaired</th>
      </tr></thead><tbody>${d.raw_readings.map(r => `<tr>
        <td style="font-family:var(--mono);font-size:11px">${esc(r.observation_id)}</td>
        <td class="num">${r.frame_number}</td>
        <td class="plate" style="color:${r.text_raw !== r.text_final ? C.warning : 'var(--ink)'}">${esc(r.text_raw)}</td>
        <td class="plate">${esc(r.text_final)}</td>
        <td class="num">${num(r.ocr_confidence, 3)}</td>
        <td class="num">${num(r.plate_confidence, 3)}</td>
        <td>${esc(r.variant)}</td>
        <td>${r.repaired ? '<span class="badge partial">repaired</span>' : ''}</td>
      </tr>`).join('')}</tbody></table></div></div>`;

  const pts = d.timeline.map(t => [t.latitude, t.longitude]);
  if (S.tmap) { S.tmap.remove(); S.tmap = null; }
  S.tmap = L.map('trajectory-map');
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { attribution: '© OpenStreetMap © CARTO' }).addTo(S.tmap);
  if (pts.length > 1) L.polyline(pts, { color: C.s2, weight: 4, opacity: .95 }).addTo(S.tmap);
  d.timeline.forEach((t, i) => L.circleMarker([t.latitude, t.longitude], {
    radius: 8, color: '#fff', weight: 2, fillColor: i === 0 ? C.s3 : C.s2, fillOpacity: .95
  }).bindPopup(`<b>${esc(d.plate)}</b><br>${esc(t.camera_id)} — ${esc(t.location || '')}<br>` +
    `${clock(t.observed_at)}<br>track #${t.local_track_id} · conf ${num(t.ocr_confidence, 3)}`).addTo(S.tmap));
  S.tmap.fitBounds(L.latLngBounds(pts).pad(0.35));
  setTimeout(() => S.tmap.invalidateSize(), 90);
};

document.getElementById('search-btn').onclick = () => {
  const v = document.getElementById('plate-input').value.trim().toUpperCase();
  if (v) trackPlate(v);
};
document.getElementById('plate-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('search-btn').click();
});

function renderSuggestions() {
  const box = document.getElementById('plate-suggestions');
  const multi = S.vehicles.filter(v => v.camera_count > 1).slice(0, 12);
  const list = (multi.length ? multi : S.vehicles.slice(0, 12));
  box.innerHTML = list.length
    ? `<span style="font-size:11.5px;color:var(--ink-muted);align-self:center">Detected plates:</span>` +
      list.map(v => `<span class="chip" onclick="trackPlate('${esc(v.plate_number)}')">${esc(v.plate_number)}</span>`).join('')
    : '';
}

/* -------------------------------------------------------- analytics */
async function loadAnalytics() {
  const [flows, stats, tl, types, s] = await Promise.all([
    api('/flows'), api('/stats'), api('/timeline'), api('/vehicle-types'), api('/summary')]);

  const totalFrames = stats.reduce((a, b) => a + (b.frames || 0), 0);
  const totalVeh = stats.reduce((a, b) => a + (b.vehicle_dets || 0), 0);
  const totalPlate = stats.reduce((a, b) => a + (b.plate_dets || 0), 0);
  const avgFps = stats.length ? stats.reduce((a, b) => a + (b.processing_fps || 0), 0) / stats.length : 0;
  document.getElementById('analytics-tiles').innerHTML = [
    ['Frames processed', totalFrames.toLocaleString(), 'across all cameras'],
    ['Vehicle detections', totalVeh.toLocaleString(), 'bounding boxes tracked'],
    ['Plate detections', totalPlate.toLocaleString(), 'plate regions found'],
    ['Processing speed', num(avgFps, 1), 'frames/second mean'],
    ['Active flows', flows.length, 'camera-to-camera links'],
    ['Mean OCR confidence', s.mean_ocr_conf ? num(s.mean_ocr_conf, 3) : '—', 'across all reads']
  ].map(([l, v, sub]) => `<div class="tile"><div class="label">${l}</div>
      <div class="value">${esc(v)}</div><div class="sub">${esc(sub)}</div></div>`).join('');

  const fbox = document.getElementById('table-flows');
  fbox.innerHTML = flows.length ? `<table><thead><tr>
      <th>Origin</th><th>Destination</th><th class="num">Vehicles</th><th class="num">Distance</th>
      <th class="num">Mean travel</th><th class="num">Mean speed</th></tr></thead><tbody>
      ${flows.map(f => `<tr><td>${esc(f.from_camera)}</td><td>${esc(f.to_camera)}</td>
        <td class="num">${f.vehicles}</td><td class="num">${num(f.road_distance_km, 2)} km</td>
        <td class="num">${num(f.mean_travel_minutes, 1)} min</td>
        <td class="num">${num(f.mean_speed_kmh, 1)} km/h</td></tr>`).join('')}</tbody></table>`
    : empty('No flows', 'Origin–destination flows need the same vehicle seen at two or more cameras.');

  // detections over time, per camera (small multiples, one colour each)
  const tbox = document.getElementById('chart-timeline');
  if (!tl.length) tbox.innerHTML = empty('No detections', 'Nothing to plot.');
  else {
    const cams = [...new Set(tl.map(r => r.camera_id))].sort();
    const palette = [C.s1, C.s2, C.s3];
    const maxD = Math.max(...tl.map(r => r.detections), 1);
    const maxB = Math.max(...tl.map(r => Number(r.bucket_s)), 1);
    const W = 560, H = 150, PL = 34, PB = 22;
    tbox.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">
      ${[0, .5, 1].map(f => `<line class="grid-line" x1="${PL}" x2="${W - 6}"
          y1="${(H - PB) - f * (H - PB - 8)}" y2="${(H - PB) - f * (H - PB - 8)}"/>
        <text class="axis-text" x="4" y="${(H - PB) - f * (H - PB - 8) + 3}">${Math.round(maxD * f)}</text>`).join('')}
      ${cams.map((cam, i) => {
        const pts = tl.filter(r => r.camera_id === cam)
          .map(r => `${PL + Number(r.bucket_s) / maxB * (W - PL - 10)},${(H - PB) - r.detections / maxD * (H - PB - 8)}`);
        return `<polyline fill="none" stroke="${palette[i % 3]}" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round" points="${pts.join(' ')}"/>`;
      }).join('')}
      <text class="axis-text" x="${PL}" y="${H - 6}">0s</text>
      <text class="axis-text" x="${W - 30}" y="${H - 6}">${maxB}s</text>
    </svg>
    <div class="map-legend" style="border:0;padding:8px 0 0">${cams.map((c, i) =>
      `<span class="legend-item"><span class="swatch line" style="background:${palette[i % 3]}"></span>${esc(c)}</span>`).join('')}</div>`;
  }

  const counts = await api('/vehicle-counts');
  const cntBox = document.getElementById('chart-counts');
  if (cntBox) {
    if (!counts.length) cntBox.innerHTML = empty('No counts', 'Process a video first.');
    else {
      const classes = ['car','motorcycle','bus','truck','bicycle'];
      const pal = { car: C.s1, motorcycle: C.s2, bus: C.s3, truck: C.warning, bicycle: C.muted };
      const max = Math.max(...counts.flatMap(r => classes.map(k => r[k] || 0)), 1);
      cntBox.innerHTML = counts.map(r => `
        <div style="margin-bottom:14px">
          <div style="font-size:12px;color:var(--ink);margin-bottom:6px">
            ${esc(r.camera_id)} — <b>${r.unique_vehicles ?? 0}</b> unique vehicles
            <span style="color:var(--ink-muted);font-size:11px">
              (each track counted once, min ${r.min_frames_threshold ?? 3} frames)</span></div>
          ${classes.filter(k => r[k]).map(k => `
            <div class="bar-row">
              <span class="name">${k}</span>
              <span class="bar-track"><span class="bar-fill"
                style="width:${(r[k] / max) * 100}%;background:${pal[k]}"></span></span>
              <span class="val">${r[k]}</span>
            </div>`).join('')}
        </div>`).join('');
    }
  }

  const ybox = document.getElementById('chart-types');
  if (!types.length) ybox.innerHTML = empty('No data', 'No vehicle classes recorded.');
  else {
    const max = Math.max(...types.map(t => t.count), 1);
    ybox.innerHTML = types.map(t => `<div class="bar-row">
      <span class="name">${esc(t.vehicle_type || 'unknown')}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${t.count / max * 100}%;background:${C.s3}"></span></span>
      <span class="val">${t.count}</span></div>`).join('');
  }

  document.getElementById('table-stats').innerHTML = stats.length ? `<table><thead><tr>
      <th>Camera</th><th class="num">Frames</th><th class="num">Vehicle dets</th><th class="num">Plate dets</th>
      <th class="num">OCR calls</th><th class="num">OCR success</th><th class="num">Wall time</th>
      <th class="num">FPS</th><th class="num">Failures</th></tr></thead><tbody>
      ${stats.map(r => `<tr><td>${esc(r.camera_id)}</td><td class="num">${r.frames}</td>
        <td class="num">${r.vehicle_dets}</td><td class="num">${r.plate_dets}</td>
        <td class="num">${r.ocr_calls}</td><td class="num">${r.ocr_success}</td>
        <td class="num">${num(r.wall_seconds, 1)}s</td><td class="num">${num(r.processing_fps, 2)}</td>
        <td class="num">${(r.ocr_engine_failures || 0) + (r.plate_detector_failures || 0)}</td>
      </tr>`).join('')}</tbody></table>` : NO_RUN;
}

/* ----------------------------------------------------------- cameras */
async function loadCameras() {
  if (!S.cameras.length) S.cameras = await api('/cameras');
  const g = document.getElementById('video-grid');
  g.innerHTML = S.cameras.length ? S.cameras.map(c => `
    <div class="video-card">
      <div class="video-head"><h3>${esc(c.camera_id)} — ${esc(c.name)}</h3>
        <p>${esc(c.road || '')} · ${num(c.latitude, 4)}, ${num(c.longitude, 4)}</p></div>
      ${c.has_video ? `<video controls preload="metadata" src="/api/video/${esc(c.camera_id)}"></video>`
                    : `<div class="empty" style="padding:32px">No annotated video on disk for this camera.</div>`}
      <div class="video-foot">
        <span><b>${c.unique_vehicles ?? 0}</b> unique vehicles</span>
        <span><b>${c.count_car ?? 0}</b> car</span>
        <span><b>${c.count_motorcycle ?? 0}</b> moto</span>
        <span><b>${c.count_bus ?? 0}</b> bus</span>
        <span><b>${c.count_truck ?? 0}</b> truck</span>
      </div>
      <div class="video-foot" style="padding-top:0;color:var(--ink-muted)">
        <span><b>${c.vehicles_with_plate}</b> plates read</span>
        <span><b>${c.vehicle_dets}</b> detections</span>
        <span><b>${num(c.processing_fps, 1)}</b> fps</span>
      </div>
    </div>`).join('') : NO_RUN;

  document.getElementById('table-cameras').innerHTML = S.cameras.length ? `<table><thead><tr>
      <th>Camera</th><th>Location</th><th>Road</th><th class="num">Latitude</th><th class="num">Longitude</th>
      <th>Direction</th><th>Capture start</th><th>Source</th><th>Video file</th></tr></thead><tbody>
      ${S.cameras.map(c => `<tr><td>${esc(c.camera_id)}</td><td>${esc(c.name)}</td>
        <td>${esc(c.road || '')}</td><td class="num">${num(c.latitude, 4)}</td>
        <td class="num">${num(c.longitude, 4)}</td><td>${esc(c.direction || '')}</td>
        <td class="time">${clock(c.capture_start)}</td>
        <td>${c.location_source === 'config'
              ? '<span class="badge complete">config</span>'
              : '<span class="badge partial">fallback</span>'}</td>
        <td style="font-family:var(--mono);font-size:11px">${esc(c.video_file)}</td></tr>`).join('')}
      </tbody></table>` : '';
}

/* ------------------------------------------------------------ alerts */
async function loadAlerts() {
  const [alerts, wl] = await Promise.all([api('/alerts'), fetch('/api/watchlist').then(r => r.json())]);
  document.getElementById('alert-count').textContent = `${alerts.length} total`;
  const box = document.getElementById('alert-list');
  box.innerHTML = alerts.length ? alerts.map(a => {
    const sev = (a.severity || '').toLowerCase() === 'high' ? 'high' : 'medium';
    const icon = a.alert_type === 'WATCHLIST_MATCH' ? '🚨' : '⚠️';
    return `<div class="alert-card">
      <div class="alert-icon ${sev}">${icon}</div>
      <div class="alert-body">
        <div class="alert-title">${esc(a.alert_type.replace(/_/g, ' '))}
          <span class="badge ${sev === 'high' ? 'dropped' : 'partial'}">${esc(a.severity)}</span>
          <span class="plate" style="margin-left:auto;cursor:pointer" onclick="trackPlate('${esc(a.plate_number)}')">${esc(a.plate_number)}</span></div>
        <div class="alert-reason">${esc(a.reason)}</div>
        <div class="alert-meta">${esc(a.camera_id)} · ${esc(a.location || '')} · ${clock(a.occurred_at)}</div>
      </div></div>`;
  }).join('') : empty('No alerts',
    'Alerts fire when a watchlisted plate is detected, or when a camera-to-camera transition is physically impossible.');

  document.getElementById('table-watchlist').innerHTML = wl.length ? `<table><thead><tr>
      <th>Plate</th><th>Note</th><th>Active</th><th>Added</th></tr></thead><tbody>
      ${wl.map(w => `<tr><td class="plate">${esc(w.plate_number)}</td><td>${esc(w.note || '')}</td>
        <td>${w.active ? '<span class="badge complete">active</span>' : '<span class="badge single">off</span>'}</td>
        <td class="time">${clock(w.created_at)}</td></tr>`).join('')}</tbody></table>`
    : `<div class="empty" style="padding:26px"><p>Watchlist is empty. Add a plate to enable watchlist alerts.</p></div>`;
}

document.getElementById('wl-add').onclick = async () => {
  const el = document.getElementById('wl-input');
  const p = el.value.trim().toUpperCase();
  if (!p) return;
  await fetch(`/api/watchlist/${encodeURIComponent(p)}`, { method: 'POST' });
  el.value = ''; loadAlerts();
};

/* -------------------------------------------------------------- data */
async function loadData() {
  const obs = await api('/observations?limit=1000');
  document.getElementById('obs-count').textContent = `${obs.length} rows`;
  document.getElementById('table-observations').innerHTML = obs.length ? `<table><thead><tr>
      <th>Plate</th><th>Camera</th><th>Timestamp</th><th class="num">Video t (s)</th>
      <th class="num">Frame</th><th class="num">Track</th><th>Type</th>
      <th class="num">OCR conf</th><th class="num">Plate conf</th><th class="num">Support</th>
      <th>Method</th><th>Format</th></tr></thead><tbody>
      ${obs.map(o => `<tr class="clickable" onclick="trackPlate('${esc(o.plate_number)}')">
        <td class="plate">${esc(o.plate_number)}</td><td>${esc(o.camera_id)}</td>
        <td class="time">${clock(o.observed_at)}</td><td class="num">${num(o.video_time_s, 2)}</td>
        <td class="num">${o.frame_number}</td><td class="num">#${o.local_track_id}</td>
        <td>${esc(o.vehicle_type || '')}</td><td class="num">${num(o.ocr_confidence, 3)}</td>
        <td class="num">${num(o.plate_detection_confidence, 3)}</td>
        <td class="num">${o.support}/${o.total_readings}</td>
        <td style="font-size:11px">${esc(o.aggregation_method || '')}</td>
        <td>${o.is_valid_format ? '<span class="badge complete">valid</span>' : '<span class="badge single">n/a</span>'}</td>
      </tr>`).join('')}</tbody></table>` : NO_RUN;

  const crops = obs.filter(o => o.evidence_crop).slice(0, 40);
  document.getElementById('evidence-grid').innerHTML = crops.length
    ? `<div class="evidence-grid">${crops.map(o => `<div class="evidence-card">
        <img loading="lazy" src="/api/evidence?path=${encodeURIComponent(o.evidence_crop)}" alt="">
        <div class="cap"><div class="p">${esc(o.plate_number)}</div>
        <div class="m">${esc(o.camera_id)} · ${hhmmss(o.observed_at)} · ${num(o.ocr_confidence, 2)}</div></div></div>`).join('')}</div>`
    : empty('No evidence crops', 'Plate crops are written while processing video.');
}

/* --------------------------------------------------------------- boot */
async function loadAll() {
  const tasks = [
    ['overview',  loadOverview,  'table-routes'],
    ['analytics', loadAnalytics, 'table-flows'],
    ['cameras',   loadCameras,   'video-grid'],
    ['alerts',    loadAlerts,    'alert-list'],
    ['data',      loadData,      'table-observations'],
  ];
  if (document.getElementById('view-map').classList.contains('active') && !S.mapLoaded) {
    S.mapLoaded = true;
    tasks.push(['map', loadMap, 'map']);
  }
  // allSettled, and each failure is confined to its own panel: a single broken
  // endpoint must not blank the rest of the dashboard.
  const results = await Promise.allSettled(tasks.map(([, fn]) => fn()));
  results.forEach((r, i) => {
    if (r.status === 'rejected') {
      const [name, , target] = tasks[i];
      console.error(`${name} failed:`, r.reason);
      const el = document.getElementById(target);
      if (el) el.innerHTML = empty('Could not load ' + name,
        esc(r.reason && r.reason.message) +
        '<br><br>Run the notebook to populate the database.');
    }
  });
  renderSuggestions();
}

(async function boot() {
  await loadHealth();
  const ok = await loadRuns();
  if (ok) loadAll();
  else {
    ['table-routes', 'chart-cameras', 'chart-status'].forEach(id => {
      const el = document.getElementById(id); if (el) el.innerHTML = NO_RUN;
    });
  }
  setInterval(loadHealth, 30000);
})();
