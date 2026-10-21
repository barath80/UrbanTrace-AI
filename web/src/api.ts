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

const KEY_STORE = 'ut_api_key'

export function getApiKey(): string {
  try { return localStorage.getItem(KEY_STORE) ?? '' } catch { return '' }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(KEY_STORE, key)
    else localStorage.removeItem(KEY_STORE)
  } catch { /* storage unavailable: the key lasts only until reload */ }
}

export async function api<T = unknown>(
  path: string, init?: RequestInit & { signal?: AbortSignal },
): Promise<T> {
  const key = getApiKey()
  const headers = new Headers(init?.headers)
  if (key) headers.set('X-API-Key', key)
  const res = await fetch(`${BASE}${path}`, { ...init, headers })
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

export interface MultiCamSlot {
  camera_id: string
  name: string
  road: string
  latitude: number
  longitude: number
  status: 'playing' | 'waiting' | 'disconnected' | 'reconnecting' | 'stopped' | 'ended' | 'error'
  source_type: string
  source_label: string
  video_name: string
  fps: number
  frame_count: number
  resolution: string
  video_time_s: number
  loops?: number
  timestamp: string
  has_feed: boolean
  error_message: string | null
  plates_detected_count?: number
  /** Plate boxes drawn on the current frame. */
  active_plate_boxes?: number
  /** Detection exceptions since the feed started; > 0 means readings are being lost. */
  detect_failures?: number
  last_detect_error?: string | null
  /** Frames shown with the previous boxes because the GPU was still busy. */
  detections_skipped?: number
  /** Frames the detector actually ran on. */
  detections_run?: number
  detect_every?: number
  /** Age of the most recent frame, or null if none has arrived. */
  seconds_since_frame?: number | null
}

export interface MultiCamDetection {
  plate: string
  confidence: number
  support: number
  total_readings: number
  agreement: number
  is_valid_format: boolean
  vehicle_type: string
  track_id: number
  camera_id: string
  location: string
  latitude: number
  longitude: number
  observed_at: string
  video_time_s: number
}

/** One camera-to-camera hop measured from real arrival times. */
export interface MultiCamLeg {
  from_camera: string
  to_camera: string
  departed: string
  arrived: string
  travel_minutes: number
  road_distance_km: number
  implied_speed_kmh: number | null
  distance_source: string
  verdict: string
  reason: string
}

/** A vehicle the live network has seen, with its reconstructed journey. */
export interface MultiCamVehicle {
  plate: string
  variants: string[]
  camera_count: number
  sighting_count: number
  route: string[]
  route_display: string
  first_seen: string
  last_seen: string
  confidence: number
  vehicle_type: string
  sightings: MultiCamDetection[]
  legs: MultiCamLeg[]
  status: 'MULTI_CAMERA' | 'SINGLE_CAMERA'
}

export interface MultiCamNetwork {
  vehicles: number
  multi_camera_vehicles: number
  sightings: number
  cameras_with_sightings: number
}

export interface MultiCamStatusResponse {
  cameras: MultiCamSlot[]
  detections: MultiCamDetection[]
  network: MultiCamNetwork
  timestamp: string
  all_connected: boolean
}

export interface MultiCamInfo {
  primary_ip: string
  local_ips: string[]
  port: number
  phone_urls: Record<string, string>
  cameras: MultiCamSlot[]
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

/* Multi-Camera 3-Feed Ingest & Status */
export const getMultiCamStatus = () => api<MultiCamStatusResponse>('/multi-camera/status')
export const getMultiCamDetections = () => api<MultiCamDetection[]>('/multi-camera/detections')
export const getMultiCamInfo = () => api<MultiCamInfo>('/multi-camera/info')
export const connectMultiCamUrl = (cameraId: string, url: string, videoName = '') =>
  api<MultiCamSlot>(
    `/multi-camera/connect-url?camera_id=${encodeURIComponent(cameraId)}&url=${encodeURIComponent(url)}&video_name=${encodeURIComponent(videoName)}`,
    { method: 'POST' },
  )
export const connectMultiCamLocal = (cameraId: string, videoFile = '') =>
  api<MultiCamSlot>(
    `/multi-camera/connect-local?camera_id=${encodeURIComponent(cameraId)}&video_file=${encodeURIComponent(videoFile)}`,
    { method: 'POST' },
  )
export const stopMultiCamStream = (cameraId: string) =>
  api<MultiCamSlot>(`/multi-camera/stop/${encodeURIComponent(cameraId)}`, { method: 'POST' })
export const stopAllMultiCamStreams = () =>
  api<{ status: string; cameras: MultiCamSlot[] }>('/multi-camera/stop-all', { method: 'POST' })
export const getMultiCamVehicles = () => api<MultiCamVehicle[]>('/multi-camera/vehicles')
export const resetMultiCamRegistry = () =>
  api<MultiCamNetwork>('/multi-camera/reset-registry', { method: 'POST' })

/* ------------------------------------------------ city network (scale) */
/* These read the continuous `net_*` tables: the live camera estate rather
   than a single batch run. Windows are in hours. */

export interface NetSummary {
  cameras: number
  active_cameras: number
  sightings: number
  unique_vehicles: number
  multi_camera_vehicles: number
  alerts: number
  mean_confidence: number | null
  mean_speed_kmh: number | null
  window_hours: number
}

export interface NetDensity {
  camera_id: string
  name: string
  road: string | null
  sector: string | null
  latitude: number
  longitude: number
  sightings: number
  unique_vehicles: number
  mean_confidence: number | null
  vehicles_per_minute: number
  last_sighting: string | null
}

export interface NetHeatPoint {
  camera_id: string
  name: string
  latitude: number
  longitude: number
  observations: number
  vehicles: number
}

export interface NetFlow {
  from_camera: string
  to_camera: string
  from_name: string | null
  to_name: string | null
  from_lat: number
  from_lon: number
  to_lat: number
  to_lon: number
  vehicles: number
  mean_travel_minutes: number | null
  mean_speed_kmh: number | null
  road_distance_km: number | null
}

export interface NetCongestion extends NetFlow {
  speed_kmh: number | null
  baseline_kmh: number | null
  speed_ratio: number | null
  travel_minutes: number | null
  baseline_travel_minutes: number | null
  baseline_samples: number
  severity: 'MODERATE' | 'HEAVY' | 'SEVERE'
}

export interface NetTrendBucket {
  bucket: string
  sightings: number
  unique_vehicles: number
  active_cameras: number
}

export interface NetSighting {
  sighting_id: string
  camera_id: string
  camera_name: string | null
  road: string | null
  sector: string | null
  plate: string
  cluster_id: string | null
  observed_at: string
  confidence: number
  vehicle_type: string | null
  track_id: number
  support: number
  total_readings: number
  is_valid_format: boolean
  latitude: number
  longitude: number
  direction: string | null
}

export interface NetLeg {
  from_camera: string
  to_camera: string
  departed: string
  arrived: string
  travel_minutes: number
  road_distance_km: number
  implied_speed_kmh: number | null
  distance_source: string
  verdict: string
  reason: string
}

/** A gap too long to be one continuous journey. */
export interface NetBreak {
  after_camera: string
  before_camera: string
  left_at: string
  returned_at: string
  gap_minutes: number
  reason: string
}

export interface NetTrajectory {
  plate: string
  sightings: NetSighting[]
  legs: NetLeg[]
  breaks: NetBreak[]
  journeys: number
  cameras: number
  first_seen: string | null
  last_seen: string | null
  /** Time actually spent travelling, summed over real legs. */
  travelling_minutes: number | null
  /** Wall-clock window the sightings span, which includes any breaks. */
  observed_span_minutes: number | null
  spellings: string[]
  window_hours: number
}

export interface NetPlateHit {
  plate: string
  sightings: number
  cameras: number
  first_seen: string
  last_seen: string
  mean_confidence: number
}

export interface NetAlert {
  id: number
  alert_type: string
  severity: string
  plate: string | null
  cluster_id: string | null
  camera_id: string | null
  camera_name: string | null
  occurred_at: string
  confidence: number | null
  reason: string
  evidence: Record<string, unknown> | null
  status: string
}

export interface NetCamera {
  camera_id: string
  name: string
  road: string
  sector: string
  latitude: number
  longitude: number
  direction: string
  source: string
  enabled: boolean
}

export interface NetCameras {
  registry: { cameras: number; enabled: number; edges: number; sectors: Record<string, number> }
  cameras: NetCamera[]
}

const w = (hours: number) => `hours=${hours}`

export const getNetSummary = (hours = 24) => api<NetSummary>(`/network/summary?${w(hours)}`)
export const getNetCameras = () => api<NetCameras>('/network/cameras')
export const getNetDensity = (hours = 24) => api<NetDensity[]>(`/network/density?${w(hours)}`)
export const getNetHeatmap = (hours = 24) => api<NetHeatPoint[]>(`/network/heatmap?${w(hours)}`)
export const getNetFlows = (hours = 24) =>
  api<NetFlow[]>(`/network/origin-destination?${w(hours)}`)
export const getNetCongestion = (hours = 24, minSamples = 1) =>
  api<NetCongestion[]>(`/network/congestion?${w(hours)}&min_samples=${minSamples}`)
export const getNetTrend = (hours = 24, bucketMinutes = 15) =>
  api<NetTrendBucket[]>(`/network/trend?${w(hours)}&bucket_minutes=${bucketMinutes}`)
export const getNetTrajectory = (plate: string, hours = 24) =>
  api<NetTrajectory>(`/network/trajectory/${encodeURIComponent(plate)}?${w(hours)}`)
export const searchNetPlates = (q: string, hours = 24) =>
  api<NetPlateHit[]>(`/network/search?q=${encodeURIComponent(q)}&${w(hours)}`)
export const getNetAlerts = (hours = 24) => api<NetAlert[]>(`/network/alerts?${w(hours)}`)
export const addNetWatchlist = (plate: string, reason = '') =>
  api(`/network/watchlist/${encodeURIComponent(plate)}?reason=${encodeURIComponent(reason)}`,
      { method: 'POST' })
export const getNetWatchlist = () =>
  api<{ plate: string; reason: string; severity: string; active: boolean }[]>('/network/watchlist')
export const getNetStatus = () => api<Record<string, unknown>>('/network/status')

/* ------------------------------------------------------------- fleet */

export interface NetCameraInput {
  camera_id: string
  name: string
  latitude: number
  longitude: number
  road?: string
  sector?: string
  direction?: string
  source?: string
  enabled?: boolean
}

export interface NetWorker {
  worker_id: string
  host: string | null
  started_at: string
  heartbeat_at: string
  age_s: number | null
  alive: boolean
  stats: {
    cameras_owned?: number
    throughput_fps?: number
    rate?: { current_fps: number; target_fps: number; cuts: number }
    capture?: { feeds: number; by_state: Record<string, number> }
  }
}

export interface NetFeed {
  camera_id: string
  name: string
  sector: string | null
  enabled: boolean
  source: string | null
  worker_id: string | null
  /** live | connecting | reconnecting | stopped | unmonitored | disabled | push-only */
  state: string
  source_fps: number | null
  resolution: string | null
  frames_read: number | null
  frames_submitted: number | null
  frames_skipped_static: number | null
  reconnects: number | null
  last_frame_at: string | null
  last_error: string | null
}

export interface NetFleet {
  workers_alive: number
  workers: number
  cameras: number
  feeds_by_state: Record<string, number>
}

export const getFleet = () => api<NetFleet>('/network/fleet')
export const getWorkers = () => api<NetWorker[]>('/network/workers')
export const getFeeds = () => api<NetFeed[]>('/network/feeds')
export const saveNetCamera = (c: NetCameraInput) =>
  api('/network/cameras', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(c),
  })
export const saveNetCamerasBulk = (cs: NetCameraInput[]) =>
  api<{ saved: number }>('/network/cameras/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cs),
  })
export const deleteNetCamera = (id: string) =>
  api(`/network/cameras/${encodeURIComponent(id)}`, { method: 'DELETE' })
export const setNetCameraEnabled = (id: string, enabled: boolean) =>
  api(`/network/cameras/${encodeURIComponent(id)}/enabled?enabled=${enabled}`, { method: 'PATCH' })
