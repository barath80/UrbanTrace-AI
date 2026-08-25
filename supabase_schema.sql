-- UrbanTrace AI - Supabase/PostgreSQL schema
-- Every pipeline execution is a `run`; all detections are scoped to a run so
-- repeated executions accumulate history instead of overwriting each other.

create extension if not exists postgis;
create extension if not exists pgcrypto;

-- ---------------------------------------------------------------- runs
create table if not exists runs (
    run_id          uuid primary key default gen_random_uuid(),
    started_at      timestamptz not null default now(),
    finished_at     timestamptz,
    device          text,
    gpu_name        text,
    vehicle_model   text,
    plate_backend   text,
    plate_model     text,
    ocr_model       text,
    camera_count    int,
    video_count     int,
    observation_count int,
    notes           text
);

-- ------------------------------------------------------------- cameras
create table if not exists cameras (
    id              bigserial primary key,
    run_id          uuid not null references runs(run_id) on delete cascade,
    camera_id       text not null,
    name            text,
    road            text,
    latitude        double precision not null,
    longitude       double precision not null,
    geom            geography(Point, 4326),
    direction       text,
    video_file      text,
    capture_start   timestamptz,
    location_source text,
    unique (run_id, camera_id)
);
create index if not exists idx_cameras_geom on cameras using gist (geom);

create table if not exists camera_edges (
    id                   bigserial primary key,
    run_id               uuid not null references runs(run_id) on delete cascade,
    source_camera        text not null,
    destination_camera   text not null,
    road_distance_km     double precision,
    expected_min_minutes double precision,
    expected_max_minutes double precision,
    road_segment         text,
    -- 'graphhopper' = real driven road; 'estimate' = great-circle x 1.25.
    -- The travel-time verdicts are computed from road_distance_km, so which
    -- one it is must be stored beside it.
    distance_source      text default 'estimate',
    typical_duration_min double precision,
    unique (run_id, source_camera, destination_camera)
);

create table if not exists videos (
    id           bigserial primary key,
    run_id       uuid not null references runs(run_id) on delete cascade,
    camera_id    text not null,
    file_name    text,
    width        int,
    height       int,
    fps          double precision,
    frame_count  int,
    duration_s   double precision,
    size_mb      double precision
);

create table if not exists camera_stats (
    id                       bigserial primary key,
    run_id                   uuid not null references runs(run_id) on delete cascade,
    camera_id                text not null,
    frames                   int,
    vehicle_dets             int,
    plate_dets               int,
    ocr_calls                int,
    ocr_success              int,
    wall_seconds             double precision,
    processing_fps           double precision,
    ocr_engine_failures      int default 0,
    plate_detector_failures  int default 0,
    -- Unique vehicles counted. Tracker ids are NOT vehicles: one vehicle
    -- fragments into several ids when the tracker loses it, so fragments are
    -- stitched back together and clutter (far-field specks, few-frame flicker)
    -- is gated out before counting. Class is an area-weighted vote over the
    -- track, so the frames where the vehicle was close decide it.
    --
    -- count_mode says WHICH count unique_vehicles is, and the two differ a lot
    -- wherever vehicles are parked in shot:
    --   'traffic'  - vehicles that travelled through the scene (or crossed the
    --                camera's counting line, when one is configured)
    --   'presence' - every distinct vehicle that appeared, parked included
    -- count_traffic/count_presence carry both totals regardless.
    unique_vehicles          int,
    count_mode               text,
    count_traffic            int,
    count_presence           int,
    count_car                int,
    count_motorcycle         int,
    count_bus                int,
    count_truck              int,
    count_bicycle            int,
    count_min_frames         int,
    count_fragments_stitched int,
    count_rejected           int,
    unique (run_id, camera_id)
);

-- -------------------------------------------------------- observations
-- The fundamental record: PLATE + TIMESTAMP + CAMERA.
create table if not exists observations (
    id                          bigserial primary key,
    run_id                      uuid not null references runs(run_id) on delete cascade,
    observation_id              text not null,
    camera_id                   text not null,
    plate_number                text not null,
    ocr_confidence              double precision,
    plate_detection_confidence  double precision,
    local_track_id              int,
    vehicle_type                text,
    frame_number                int,
    video_time_s                double precision,   -- MEASURED from frame index / fps
    observed_at                 timestamptz,        -- DERIVED: capture start + video_time_s
    clock_source                text,
    latitude                    double precision,
    longitude                   double precision,
    geom                        geography(Point, 4326),
    direction                   text,
    aggregation_method          text,
    support                     int,
    total_readings              int,
    agreement                   double precision,
    is_valid_format             boolean,
    evidence_crop               text,
    unique (run_id, observation_id)
);
create index if not exists idx_obs_plate   on observations (plate_number);
create index if not exists idx_obs_camera  on observations (run_id, camera_id);
create index if not exists idx_obs_time    on observations (observed_at);
create index if not exists idx_obs_geom    on observations using gist (geom);

-- Raw per-frame OCR: the audit trail behind every aggregated plate.
create table if not exists plate_readings (
    id                bigserial primary key,
    run_id            uuid not null references runs(run_id) on delete cascade,
    observation_id    text not null,
    camera_id         text,
    local_track_id    int,
    frame_number      int,
    timestamp_s       double precision,
    text_raw          text,          -- exactly what OCR emitted
    text_final        text,          -- after Indian-plate grammar validation
    ocr_confidence    double precision,
    plate_confidence  double precision,
    is_valid_format   boolean,
    repaired          boolean,
    variant           text
);
create index if not exists idx_readings_obs on plate_readings (run_id, observation_id);

-- ----------------------------------------------------- global vehicles
create table if not exists global_vehicles (
    id                     bigserial primary key,
    run_id                 uuid not null references runs(run_id) on delete cascade,
    global_vehicle_id      text not null,
    plate_number           text not null,
    route                  text[],
    route_display          text,
    route_status           text,
    dropped_at             text,
    status_reason          text,
    first_seen             timestamptz,
    last_seen              timestamptz,
    total_duration_minutes double precision,
    camera_count           int,
    confidence             double precision,
    merged_variants        text[],
    unique (run_id, global_vehicle_id)
);
create index if not exists idx_gv_plate on global_vehicles (plate_number);

create table if not exists vehicle_observations (
    id                bigserial primary key,
    run_id            uuid not null references runs(run_id) on delete cascade,
    global_vehicle_id text not null,
    observation_id    text not null,
    seq               int
);

create table if not exists route_legs (
    id                   bigserial primary key,
    run_id               uuid not null references runs(run_id) on delete cascade,
    global_vehicle_id    text not null,
    leg_index            int,
    from_camera          text,
    to_camera            text,
    departed             timestamptz,
    arrived              timestamptz,
    travel_minutes       double precision,
    road_distance_km     double precision,
    implied_speed_kmh    double precision,
    expected_min_minutes double precision,
    expected_max_minutes double precision,
    distance_source      text default 'estimate',
    typical_duration_min double precision,
    verdict              text,
    reason               text
);
create index if not exists idx_legs_vehicle on route_legs (run_id, global_vehicle_id);

-- -------------------------------------------------------------- alerts
create table if not exists alerts (
    id             bigserial primary key,
    run_id         uuid references runs(run_id) on delete cascade,
    alert_type     text not null,
    severity       text,
    plate_number   text,
    camera_id      text,
    location       text,
    occurred_at    timestamptz,
    ocr_confidence double precision,
    evidence       text,
    reason         text,
    status         text default 'NEW',
    created_at     timestamptz default now()
);
create index if not exists idx_alerts_plate on alerts (plate_number);

create table if not exists watchlist (
    id           bigserial primary key,
    plate_number text unique not null,
    note         text,
    active       boolean default true,
    created_at   timestamptz default now()
);

-- --------------------------------------------------------------- views
-- Per-camera vehicle counts, straight from stored detections.
create or replace view v_camera_counts as
select o.run_id,
       o.camera_id,
       c.name                              as location,
       count(distinct o.plate_number)      as vehicles_with_plate,
       count(*)                            as observations,
       round(avg(o.ocr_confidence)::numeric, 3) as mean_ocr_confidence,
       min(o.observed_at)                  as first_seen,
       max(o.observed_at)                  as last_seen
from observations o
left join cameras c on c.run_id = o.run_id and c.camera_id = o.camera_id
group by o.run_id, o.camera_id, c.name;

-- Camera-to-camera flow (origin-destination) from reconstructed routes.
create or replace view v_camera_flows as
select run_id, from_camera, to_camera,
       count(*)                                   as vehicles,
       round(avg(travel_minutes)::numeric, 2)     as mean_travel_minutes,
       round(avg(implied_speed_kmh)::numeric, 1)  as mean_speed_kmh,
       max(road_distance_km)                      as road_distance_km
from route_legs
where verdict <> 'IMPLAUSIBLE'
group by run_id, from_camera, to_camera;

-- Unique vehicles counted per camera, by class. Dropped rather than replaced:
-- CREATE OR REPLACE VIEW cannot rename or reorder existing columns, so an
-- older database re-running this file would fail on the added columns.
drop view if exists v_vehicle_counts;
create view v_vehicle_counts as
select run_id, camera_id, unique_vehicles, count_mode,
       count_traffic    as traffic_count,
       count_presence   as presence_count,
       count_car        as car,
       count_motorcycle as motorcycle,
       count_bus        as bus,
       count_truck      as truck,
       count_bicycle    as bicycle,
       count_min_frames as min_frames_threshold,
       count_fragments_stitched as fragments_stitched,
       count_rejected   as rejected_as_clutter
from camera_stats
order by camera_id;

-- Full trajectory timeline for any plate.
drop view if exists v_plate_timeline cascade;
create view v_plate_timeline as
select o.run_id, o.observation_id, o.plate_number, o.observed_at, o.video_time_s,
       o.camera_id, c.name as location, c.latitude, c.longitude,
       o.local_track_id, o.ocr_confidence, o.vehicle_type, o.evidence_crop
from observations o
left join cameras c on c.run_id = o.run_id and c.camera_id = o.camera_id
order by o.plate_number, o.observed_at;

-- Every sighting of a VEHICLE, including sightings recorded under a merged OCR
-- variant of its plate. Matching observations on plate_number alone silently
-- drops those, which understates the trajectory.
drop view if exists v_vehicle_timeline;
create view v_vehicle_timeline as
select vo.run_id, vo.global_vehicle_id, g.plate_number as vehicle_plate,
       vo.seq, t.observation_id, t.plate_number, t.observed_at, t.video_time_s,
       t.camera_id, t.location, t.latitude, t.longitude,
       t.local_track_id, t.ocr_confidence, t.vehicle_type, t.evidence_crop,
       (t.plate_number is distinct from g.plate_number) as is_merged_variant
from vehicle_observations vo
join global_vehicles g
  on g.run_id = vo.run_id and g.global_vehicle_id = vo.global_vehicle_id
join v_plate_timeline t
  on t.run_id = vo.run_id and t.observation_id = vo.observation_id
order by g.plate_number, t.observed_at;
