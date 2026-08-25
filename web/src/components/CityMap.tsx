/**
 * The GIS surface: MapLibre GL, tilted, with real 3D building extrusions.
 *
 * Everything drawn here is data from the API. The three layer families are
 * visually distinct on purpose, because they have different epistemic status:
 *
 *   heat   - MEASURED presence. Where plates were actually read.
 *   flows  - real road geometry between cameras, solid when routed, DASHED
 *            when the backend fell back to a straight-line estimate.
 *   route  - one vehicle's path. Animated in the direction of travel, and
 *            dashed if it too is an estimate.
 *
 * A dashed line therefore always means "this is not a surveyed road", and the
 * legend says so. Nothing here is decorative.
 */
import * as maplibregl from 'maplibre-gl'
import type { LngLatBoundsLike, Map as MLMap, MapMouseEvent } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
// MapLibre parses tiles in a web worker. Left alone, it asks for a sibling
// module at runtime that the bundle never emits, and the map renders blank
// with no console error. Bundling the worker explicitly and handing MapLibre
// its real URL is the supported fix.
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useRef } from 'react'

maplibregl.setWorkerUrl(workerUrl)

import type { Camera, HeatPoint } from '../api'
import type { Feature, FeatureCollection } from 'geojson'

// CARTO's dark vector basemap: free, keyless, and it ships a building layer we
// can extrude for the 3D view.
const STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

export interface FlowLine {
  from: string
  to: string
  coordinates: [number, number][]   // [lat, lon]
  weight: number                    // 0..1, drives thickness
  real: boolean
  popup: string
}

export interface RouteLine {
  coordinates: [number, number][]   // [lat, lon]
  real: boolean
  stops: { lat: number; lon: number; label: string; popup: string }[]
}

export interface CityMapProps {
  cameras?: Camera[]
  heat?: HeatPoint[] | [number, number, number][]
  flows?: FlowLine[]
  route?: RouteLine | null
  showHeat?: boolean
  showFlows?: boolean
  showCameras?: boolean
  pitch?: number
  className?: string
  onPickCamera?: (id: string) => void
}

/** [lat, lon] pairs -> GeoJSON [lon, lat]. Getting this backwards silently
    puts Chennai in the Indian Ocean, so it happens in exactly one place. */
const toLngLat = (c: [number, number][]) => c.map(([lat, lon]) => [lon, lat])

function heatFeatures(heat: CityMapProps['heat']): FeatureCollection {
  const feats: Feature[] = []
  for (const h of heat ?? []) {
    if (Array.isArray(h)) {
      const [lat, lon, w] = h
      feats.push({
        type: 'Feature', properties: { w },
        geometry: { type: 'Point', coordinates: [lon, lat] },
      })
    } else {
      feats.push({
        type: 'Feature', properties: { w: h.observations || 1 },
        geometry: { type: 'Point', coordinates: [h.longitude, h.latitude] },
      })
    }
  }
  return { type: 'FeatureCollection', features: feats }
}

export default function CityMap({
  cameras = [], heat = [], flows = [], route = null,
  showHeat = true, showFlows = true, showCameras = true,
  pitch = 45, className = 'map-canvas', onPickCamera,
}: CityMapProps) {
  const host = useRef<HTMLDivElement>(null)
  const map = useRef<MLMap | null>(null)
  const ready = useRef(false)
  const markers = useRef<maplibregl.Marker[]>([])
  const dash = useRef<number>(0)
  // The map finishes loading asynchronously, so the draw routine is kept in a
  // ref: whichever happens last - the data arriving or the style loading -
  // calls the current version. Without this the first paint is silently
  // skipped and the map renders as an empty basemap.
  const draw = useRef<() => void>(() => {})

  /* ---- create once ---- */
  useEffect(() => {
    if (!host.current || map.current) return
    const m = new maplibregl.Map({
      container: host.current,
      style: STYLE,
      center: [80.17, 13.09],
      zoom: 11,
      pitch,
      bearing: -12,
      attributionControl: { compact: true },
    })
    map.current = m
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
    m.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left')

    m.on('load', () => {
      ready.current = true

      // 3D buildings, where the basemap has footprints.
      if (!m.getLayer('ut-buildings')) {
        const src = m.getStyle().layers?.find((l: { id: string }) => l.id.includes('building'))
        if (src) {
          m.addLayer({
            id: 'ut-buildings', type: 'fill-extrusion', source: 'carto',
            'source-layer': 'building', minzoom: 13,
            paint: {
              'fill-extrusion-color': '#1d2431',
              'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 12],
              'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
              'fill-extrusion-opacity': 0.72,
            },
          })
        }
      }

      m.addSource('ut-heat', { type: 'geojson', data: heatFeatures([]) })
      m.addLayer({
        id: 'ut-heat', type: 'heatmap', source: 'ut-heat',
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'w'], 0, 0, 1, 1],
          'heatmap-intensity': 1.1,
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 9, 18, 15, 52],
          'heatmap-opacity': 0.72,
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0, 'rgba(0,0,0,0)',
            0.2, 'rgba(25,158,112,.55)',
            0.45, 'rgba(250,178,25,.65)',
            0.7, 'rgba(236,131,90,.78)',
            1, 'rgba(208,59,59,.9)',
          ],
        },
      })

      m.addSource('ut-flows', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({
        id: 'ut-flows', type: 'line', source: 'ut-flows',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#3987e5',
          'line-width': ['interpolate', ['linear'], ['get', 'weight'], 0, 2, 1, 10],
          'line-opacity': 0.85,
          'line-dasharray': ['case', ['get', 'real'], ['literal', [1, 0]], ['literal', [2, 1.6]]],
        },
      })

      m.addSource('ut-route', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({
        id: 'ut-route-case', type: 'line', source: 'ut-route',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#000', 'line-width': 11, 'line-opacity': 0.5 },
      })
      m.addLayer({
        id: 'ut-route', type: 'line', source: 'ut-route',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#d95926', 'line-width': 5.5, 'line-opacity': 0.98 },
      })
      // A travelling dash reads as direction of travel without an animation
      // library, and stops entirely when the user prefers reduced motion.
      m.addLayer({
        id: 'ut-route-flow', type: 'line', source: 'ut-route',
        layout: { 'line-cap': 'butt' },
        paint: {
          'line-color': '#ffd9c6', 'line-width': 2.4, 'line-opacity': 0.9,
          'line-dasharray': [0, 4, 3],
        },
      })

      m.on('click', 'ut-flows', (e: MapMouseEvent & { features?: Feature[] }) => {
        const f = e.features?.[0]
        if (!f) return
        new maplibregl.Popup({ closeButton: false })
          .setLngLat(e.lngLat).setHTML(String(f.properties?.popup ?? '')).addTo(m)
      })
      m.on('mouseenter', 'ut-flows', () => { m.getCanvas().style.cursor = 'pointer' })
      m.on('mouseleave', 'ut-flows', () => { m.getCanvas().style.cursor = '' })

      draw.current()
    })

    // 'load' can fire before glyphs and sprites settle; redraw once idle so a
    // first paint that landed too early is corrected.
    m.once('idle', () => draw.current())

    return () => { m.remove(); map.current = null; ready.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ---- animate the route dash ---- */
  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduce || !route) return
    let raf = 0
    const seq = [[0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5], [2, 4, 1],
                 [2.5, 4, 0.5], [3, 4, 0], [0, 0.5, 3, 3.5], [0, 1, 3, 3],
                 [0, 1.5, 3, 2.5], [0, 2, 3, 2], [0, 2.5, 3, 1.5], [0, 3, 3, 1],
                 [0, 3.5, 3, 0.5]]
    let last = 0
    const tick = (t: number) => {
      if (t - last > 55) {
        last = t
        dash.current = (dash.current + 1) % seq.length
        const m = map.current
        if (m?.getLayer('ut-route-flow')) {
          m.setPaintProperty('ut-route-flow', 'line-dasharray', seq[dash.current])
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [route])

  /* ---- data updates ---- */
  useEffect(() => {
    const m = map.current
    if (!m) return

    const apply = () => {
      // Only our own sources matter here, and they exist from 'load' onward.
      // isStyleLoaded() can still report false at that point, so gating on it
      // would drop the first paint entirely.
      if (!m.getSource('ut-heat')) return

      ;(m.getSource('ut-heat') as maplibregl.GeoJSONSource).setData(heatFeatures(heat))
      m.setLayoutProperty('ut-heat', 'visibility', showHeat ? 'visible' : 'none')

      ;(m.getSource('ut-flows') as maplibregl.GeoJSONSource).setData({
        type: 'FeatureCollection',
        features: flows.map(f => ({
          type: 'Feature' as const,
          properties: { weight: f.weight, real: f.real, popup: f.popup },
          geometry: { type: 'LineString' as const, coordinates: toLngLat(f.coordinates) },
        })),
      })
      for (const id of ['ut-flows']) {
        m.setLayoutProperty(id, 'visibility', showFlows ? 'visible' : 'none')
      }

      const routeFeats: Feature[] = route && route.coordinates.length > 1
        ? [{
            type: 'Feature', properties: { real: route.real },
            geometry: { type: 'LineString', coordinates: toLngLat(route.coordinates) },
          }]
        : []
      ;(m.getSource('ut-route') as maplibregl.GeoJSONSource).setData({
        type: 'FeatureCollection', features: routeFeats,
      })
      for (const id of ['ut-route', 'ut-route-case', 'ut-route-flow']) {
        if (m.getLayer(id)) {
          m.setPaintProperty(id, 'line-dasharray',
            id === 'ut-route-flow' ? [0, 4, 3]
              : route?.real === false ? [2, 1.4] : [1, 0])
        }
      }

      // Markers are DOM, not layers, so they are rebuilt wholesale.
      markers.current.forEach(mk => mk.remove())
      markers.current = []

      if (showCameras) {
        for (const c of cameras) {
          const el = document.createElement('div')
          el.className = 'pin cam'
          el.title = `${c.camera_id} — ${c.name}`
          const mk = new maplibregl.Marker({ element: el })
            .setLngLat([c.longitude, c.latitude])
            .setPopup(new maplibregl.Popup({ closeButton: false, offset: 14 }).setHTML(
              `<b>${c.camera_id}</b> — ${c.name}<br>${c.road ?? ''}<br>` +
              `${c.latitude.toFixed(4)}, ${c.longitude.toFixed(4)}<hr style="border:0;border-top:1px solid #2b3543;margin:6px 0">` +
              `${c.vehicles_with_plate} vehicles identified<br>` +
              `${c.unique_vehicles} unique vehicles counted (${c.count_mode ?? 'traffic'})<br>` +
              `${c.vehicle_dets} detections · ${c.plate_dets} plate detections<br>` +
              `location source: ${c.location_source}`))
            .addTo(m)
          if (onPickCamera) el.addEventListener('click', () => onPickCamera(c.camera_id))
          markers.current.push(mk)
        }
      }

      route?.stops.forEach(s => {
        const el = document.createElement('div')
        el.className = 'pin stop pulse'
        el.textContent = s.label
        markers.current.push(
          new maplibregl.Marker({ element: el })
            .setLngLat([s.lon, s.lat])
            .setPopup(new maplibregl.Popup({ closeButton: false, offset: 18 }).setHTML(s.popup))
            .addTo(m),
        )
      })

      // Frame whatever is most specific: a selected route, else the network.
      const pts: [number, number][] = route && route.coordinates.length > 1
        ? route.coordinates
        : cameras.map(c => [c.latitude, c.longitude])
      if (pts.length === 1) {
        m.easeTo({ center: [pts[0][1], pts[0][0]], zoom: 14, duration: 700 })
      } else if (pts.length > 1) {
        const lats = pts.map(p => p[0]); const lons = pts.map(p => p[1])
        const b: LngLatBoundsLike = [
          [Math.min(...lons), Math.min(...lats)],
          [Math.max(...lons), Math.max(...lats)],
        ]
        m.fitBounds(b, { padding: 90, duration: 900, maxZoom: 15 })
      }
    }

    draw.current = apply
    if (ready.current) apply()
  }, [cameras, heat, flows, route, showHeat, showFlows, showCameras, onPickCamera])

  useEffect(() => {
    const m = map.current
    if (m && ready.current) m.easeTo({ pitch, duration: 600 })
  }, [pitch])

  return <div ref={host} className={className} />
}
