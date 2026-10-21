-- UrbanTrace AI -- continuous city-scale schema.
--
-- The batch schema scopes everything to a `run`, which suits reprocessing a
-- folder of videos. A live city network has no runs: it has a continuous
-- stream of sightings from a fixed set of cameras. These tables model that,
-- and coexist with the batch tables rather than replacing them.
--
-- Sizing note: a 1000-camera network reading ~1 plate/camera/second produces
-- on the order of 86 million sightings a day. Two things follow, and both are
-- reflected below:
--   * sightings are RANGE PARTITIONED by day, so retention is a partition drop
--     (instant) rather than a DELETE over tens of millions of rows;
--   * every query the dashboard issues is served by an index, because a
--     sequential scan of a table that size is not a query, it is an outage.

create extension if not exists postgis;
create extension if not exists pgcrypto;

-- ------------------------------------------------------------- cameras
-- Persistent camera estate. Not scoped to a run: a camera exists whether or
-- not anything has been processed today.
create table if not exists net_cameras (
    camera_id     text primary key,
    site_id       text not null default 'default',
    name          text not null,
    road          text,
    sector        text,
    latitude      double precision not null,
    longitude     double precision not null,
    geom          geography(Point, 4326)
                  generated always as (
                      st_setsrid(st_makepoint(longitude, latitude), 4326)::geography
                  ) stored,
    direction     text,
    source        text,
    enabled       boolean not null default true,
    installed_at  timestamptz default now(),
    updated_at    timestamptz default now(),
    constraint net_cameras_lat_ck check (latitude between -90 and 90),
    constraint net_cameras_lon_ck check (longitude between -180 and 180)
);
create index if not exists idx_net_cameras_geom   on net_cameras using gist (geom);
create index if not exists idx_net_cameras_sector on net_cameras (sector);
create index if not exists idx_net_cameras_site   on net_cameras (site_id);

-- Plausible road links, k-nearest rather than all-pairs.
create table if not exists net_edges (
    source_camera        text not null references net_cameras(camera_id) on delete cascade,
    destination_camera   text not null references net_cameras(camera_id) on delete cascade,
    road_distance_km     double precision not null,
    expected_min_minutes double precision not null,
    expected_max_minutes double precision not null,
    distance_source      text not null default 'estimate',
    primary key (source_camera, destination_camera)
);

-- ---------------------------------------------------------- sightings
-- One confirmed plate reading, at one camera, at one instant. This is the
-- fundamental fact of the whole platform; everything else is derived from it.
create table if not exists net_sightings (
    id             bigint generated always as identity,
    sighting_id    uuid not null default gen_random_uuid(),
    site_id        text not null default 'default',
    camera_id      text not null,
    plate          text not null,
    plate_canonical text not null,        -- OCR-confusion-folded, for blocking
    cluster_id     text,                  -- assigned vehicle identity
    observed_at    timestamptz not null,
    ingested_at    timestamptz not null default now(),
    confidence     double precision not null,
    support        int,
    total_readings int,
    is_valid_format boolean,
    vehicle_type   text,
    track_id       bigint,
    latitude       double precision,
    longitude      double precision,
    direction      text,
    evidence_crop  text,
    primary key (id, observed_at)
) partition by range (observed_at);

-- Indexes are declared on the partitioned parent so every partition inherits
-- them. The three access paths that matter:
--   plate lookup  -> trajectory reconstruction for one vehicle
--   camera+time   -> per-camera density and flow analytics
--   cluster       -> assembling a vehicle's journey
create index if not exists idx_net_sight_plate    on net_sightings (plate, observed_at desc);
create index if not exists idx_net_sight_canon    on net_sightings (plate_canonical, observed_at desc);
create index if not exists idx_net_sight_cam_time on net_sightings (camera_id, observed_at desc);
create index if not exists idx_net_sight_cluster  on net_sightings (cluster_id, observed_at);
create index if not exists idx_net_sight_time     on net_sightings (observed_at desc);

-- A default partition means an insert can never fail because tomorrow's
-- partition was not created yet. `ensure_sighting_partition` moves data into
-- properly bounded daily partitions.
create table if not exists net_sightings_default
    partition of net_sightings default;

-- Create the daily partition covering `day`, if absent.
create or replace function ensure_sighting_partition(day date)
returns void language plpgsql as $$
declare
    part text := format('net_sightings_%s', to_char(day, 'YYYYMMDD'));
begin
    if to_regclass(part) is null then
        execute format(
            'create table %I partition of net_sightings for values from (%L) to (%L)',
            part, day, day + 1);
    end if;
end $$;

-- ----------------------------------------------------------- vehicles
-- A resolved identity: the set of sightings judged to be one physical vehicle.
create table if not exists net_vehicles (
    cluster_id     text primary key,
    site_id        text not null default 'default',
    plate          text not null,
    spellings      jsonb not null default '{}'::jsonb,
    first_seen     timestamptz not null,
    last_seen      timestamptz not null,
    camera_count   int not null default 0,
    sighting_count int not null default 0,
    confidence     double precision,
    vehicle_type   text,
    updated_at     timestamptz not null default now()
);
create index if not exists idx_net_veh_plate on net_vehicles (plate);
create index if not exists idx_net_veh_last  on net_vehicles (last_seen desc);
create index if not exists idx_net_veh_cams  on net_vehicles (camera_count desc);

-- One measured camera-to-camera hop.
create table if not exists net_legs (
    id                bigint generated always as identity primary key,
    cluster_id        text not null,
    seq               int not null,
    from_camera       text not null,
    to_camera         text not null,
    departed          timestamptz not null,
    arrived           timestamptz not null,
    travel_minutes    double precision not null,
    road_distance_km  double precision not null,
    implied_speed_kmh double precision,
    distance_source   text,
    verdict           text not null,
    reason            text,
    unique (cluster_id, seq)
);
create index if not exists idx_net_legs_pair on net_legs (from_camera, to_camera, arrived desc);
create index if not exists idx_net_legs_time on net_legs (arrived desc);

-- ------------------------------------------------------------- alerts
create table if not exists net_watchlist (
    plate       text primary key,
    reason      text,
    severity    text not null default 'HIGH',
    active      boolean not null default true,
    added_at    timestamptz not null default now()
);

create table if not exists net_alerts (
    id           bigint generated always as identity primary key,
    site_id      text not null default 'default',
    alert_type   text not null,
    severity     text not null default 'MEDIUM',
    plate        text,
    cluster_id   text,
    camera_id    text,
    occurred_at  timestamptz not null,
    created_at   timestamptz not null default now(),
    confidence   double precision,
    reason       text not null,
    evidence     jsonb,
    status       text not null default 'NEW'
);
create index if not exists idx_net_alerts_time  on net_alerts (occurred_at desc);
create index if not exists idx_net_alerts_plate on net_alerts (plate);
create index if not exists idx_net_alerts_type  on net_alerts (alert_type, occurred_at desc);

-- ---------------------------------------------------- rolling analytics
-- Per-camera, per-minute counts. Aggregating raw sightings on every dashboard
-- refresh would scan the whole day; this table is written once per minute per
-- camera and every density/heatmap query reads it instead.
create table if not exists net_camera_minute (
    camera_id    text not null,
    bucket       timestamptz not null,      -- truncated to the minute
    sightings    int not null default 0,
    unique_plates int not null default 0,
    mean_confidence double precision,
    primary key (camera_id, bucket)
);
create index if not exists idx_net_min_bucket on net_camera_minute (bucket desc);

-- Baseline speed per camera pair and time-of-day slot, used to decide whether
-- a corridor is congested RELATIVE TO ITS OWN NORMAL rather than against a
-- fixed number that means nothing on a given road.
create table if not exists net_flow_baseline (
    from_camera   text not null,
    to_camera     text not null,
    hour_of_day   int  not null check (hour_of_day between 0 and 23),
    samples       int  not null default 0,
    mean_speed_kmh double precision,
    mean_travel_minutes double precision,
    updated_at    timestamptz not null default now(),
    primary key (from_camera, to_camera, hour_of_day)
);

-- ------------------------------------------------------------ cluster
-- Ingest workers heartbeat here. A worker whose heartbeat is older than the
-- TTL is treated as gone and its cameras are taken over by the others.
create table if not exists net_workers (
    worker_id    text primary key,
    host         text,
    started_at   timestamptz not null default now(),
    heartbeat_at timestamptz not null default now(),
    stats        jsonb not null default '{}'::jsonb
);
create index if not exists idx_net_workers_beat on net_workers (heartbeat_at desc);

-- Latest state of every feed, written by the worker that owns it.
create table if not exists net_camera_health (
    camera_id          text primary key,
    worker_id          text,
    state              text not null,
    source_fps         double precision,
    resolution         text,
    frames_read        bigint not null default 0,
    frames_submitted   bigint not null default 0,
    frames_skipped_static bigint not null default 0,
    reconnects         int not null default 0,
    last_frame_at      timestamptz,
    last_error         text,
    updated_at         timestamptz not null default now()
);
create index if not exists idx_net_health_state on net_camera_health (state);
create index if not exists idx_net_health_worker on net_camera_health (worker_id);
