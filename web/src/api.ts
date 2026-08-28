/**
 * Typed client for the UrbanTrace API.
 *
 * Every value rendered by this app comes from these endpoints, which read the
 * pipeline's Supabase tables. There are no fixtures and no fallback constants:
 * when the database is empty the endpoints return empty collections and the UI
 * says so, rather than showing something that looks like data.
 */

const BASE = '/api'

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

export async function api<T = unknown>(
  path: string, init?: RequestInit & { signal?: AbortSignal },
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json())?.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(String(detail), res.status)
  }
  return res.json() as Promise<T>
}

/* ----------------------------------------------------------- shapes */

export interface Health {
  runtime: { device: string; cuda: boolean; gpu_name?: string; torch: string }
  database: { connected: boolean; server?: string; public_tables?: number; error?: string }
  latest_run: string | null
}

export interface Camera {
  camera_id: string
  name: string
  road: string | null
  latitude: number
  longitude: number
  direction: string
  video_file: string
  capture_start: string
  location_source: string
  frames: number
  vehicle_dets: number
  plate_dets: number
  processing_fps: number
  unique_vehicles: number
  count_mode: string | null
  count_traffic: number
  count_presence: number
  count_car: number
  count_motorcycle: number
  count_bus: number
  count_truck: number
  vehicles_with_plate: number
  has_video: boolean
}

export interface Summary {
  run_id: string
  observations: number
  unique_plates: number
  multi_camera: number
  alerts: number
  cameras: number
  avg_speed_kmh: number | null
  avg_travel_min: number | null
  mean_ocr_conf: number | null
  raw_readings: number
  [k: string]: unknown
}

export interface Sighting {
  observation_id: string
  camera_id: string
  plate_number: string
  observed_at: string
  video_time_s: number
  location: string | null
  latitude: number
  longitude: number
  local_track_id: number
  ocr_confidence: number
  vehicle_type: string | null
  evidence_crop: string | null
  is_merged_variant?: boolean
}

export interface Leg {
  from_camera: string
  to_camera: string
  departed: string
  arrived: string
  travel_minutes: number
  road_distance_km: number
  implied_speed_kmh: number
  expected_min_minutes: number
  expected_max_minutes: number
  distance_source: string
  typical_duration_min: number | null
  verdict: string
  reason: string
}

export interface RoadLeg {
  from_camera: string
  to_camera: string
  distance_km: number
  typical_duration_min: number
  source: string
  is_real_road: boolean
  error: string | null
  coordinates: [number, number][]
}

export interface VehiclePath {
  global_vehicle_id: string
  plate_number: string
  route: string[]
  points: { camera_id: string; name: string; latitude: number; longitude: number }[]
  legs: RoadLeg[]
  coordinates: [number, number][]
  road_distance_km: number
  typical_duration_min: number
  measured_duration_min: number | null
  is_real_road: boolean
  path_note: string
}

export interface JourneyStats {
  sightings: number
  cameras: number
  measured_duration_min: number | null
  road_distance_km: number | null
  typical_duration_min: number | null
  delay_vs_typical_min: number | null
  mean_implied_speed_kmh: number | null
  verdicts: Record<string, number>
  provenance: Record<string, string>
}

export interface Alert {
  alert_type: string
  severity: string
  camera_id: string
  location: string
  occurred_at: string
  ocr_confidence: number
  evidence: string | null
  reason: string
  status: string
  plate_number?: string
}

export interface Reading {
  observation_id: string
  frame_number: number
  text_raw: string
  text_final: string
  ocr_confidence: number
  plate_confidence: number
  variant: string
  repaired: boolean
}

export interface Dossier {
  plate: string
  vehicle: Record<string, unknown> | null
  timeline: Sighting[]
  legs: Leg[]
  path: VehiclePath | null
  heat: [number, number, number][]
  stats: JourneyStats
  alerts: Alert[]
  raw_readings: Reading[]
}

export interface Flow {
  from_camera: string
  to_camera: string
  vehicles: number
  mean_travel_minutes: number
  mean_speed_kmh: number
  road_distance_km: number
}

export interface HeatPoint {
  camera_id: string
  name: string
  latitude: number
  longitude: number
  observations: number
  vehicles: number
}

export interface RoutingStatus {
  provider: string
  api_key_configured: boolean
  cached_routes: number
  fallback: string
}

export interface VehicleRow {
  plate_number: string
  route_display: string
  route_status: string
  camera_count: number
  first_seen: string
  last_seen: string
  confidence: number
  [k: string]: unknown
}

export interface TimelineBucket {
  /** Start of the bucket in VIDEO time, seconds from the clip start.
      The endpoint buckets on video_time_s, which is the measured quantity —
      wall-clock is derived, so it would be the wrong axis for a trend. */
  bucket_s: number
  camera_id: string
  detections: number
}

/* --------------------------------------------------------- endpoints */

export const getHealth = () => api<Health>('/health')
export const getSummary = () => api<Summary>('/summary')
export const getCameras = () => api<Camera[]>('/cameras')
export const getFlows = () => api<Flow[]>('/flows')
export const getHeatmap = () => api<HeatPoint[]>('/heatmap')
export const getRouting = () => api<RoutingStatus>('/routing')
export const getVehicles = () => api<VehicleRow[]>('/vehicles')
export const getVehiclePaths = () => api<Record<string, VehiclePath>>('/vehicle-paths')
export const getAlerts = () => api<Alert[]>('/alerts')
export const getVehicleCounts = () => api<Record<string, unknown>[]>('/vehicle-counts')
export const getVehicleTypes = () => api<Record<string, unknown>[]>('/vehicle-types')
export const getTimeline = () => api<TimelineBucket[]>('/timeline')
export const getObservations = (limit = 500) =>
  api<Record<string, unknown>[]>(`/observations?limit=${limit}`)
export const getDossier = (plate: string) =>
  api<Dossier>(`/vehicle-dossier/${encodeURIComponent(plate)}`)
export const searchPlates = (q: string) =>
  api<VehicleRow[]>(`/search?q=${encodeURIComponent(q)}`)
/* Live ingest is control, not query, so these are POSTs. */
export const liveStart = (cameraId: string, detectEvery: number) =>
  api(`/live/start?camera_id=${encodeURIComponent(cameraId)}` +
      `&detect_every=${detectEvery}`, { method: 'POST' })
export const liveStop = (cameraId?: string) =>
  api(`/live/stop${cameraId ? `?camera_id=${encodeURIComponent(cameraId)}` : ''}`,
      { method: 'POST' })

export const roadRoute = (from: string, to: string) =>
  api<RoadLeg & { is_real_road: boolean }>(
    `/road-route?from_camera=${encodeURIComponent(from)}&to_camera=${encodeURIComponent(to)}`,
  )
