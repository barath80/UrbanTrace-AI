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

-- Full trajectory timeline for any plate.
create or replace view v_plate_timeline as
select o.run_id, o.plate_number, o.observed_at, o.video_time_s,
       o.camera_id, c.name as location, c.latitude, c.longitude,
       o.local_track_id, o.ocr_confidence, o.vehicle_type, o.evidence_crop
from observations o
left join cameras c on c.run_id = o.run_id and c.camera_id = o.camera_id
order by o.plate_number, o.observed_at;
