-- Real road distances from GraphHopper.
--
-- Camera-to-camera distance used to be great-circle x 1.25 - a guess that the
-- PLAUSIBLE / FAST / SLOW travel-time verdicts were computed from. It is now
-- the real driven road where routing is available. Storing distance_source
-- alongside the number keeps an estimate from ever being read back as a
-- measurement.
--
-- Additive and idempotent: safe to run on an existing database.

alter table camera_edges add column if not exists distance_source     text default 'estimate';
alter table camera_edges add column if not exists typical_duration_min double precision;

alter table route_legs   add column if not exists distance_source     text default 'estimate';
alter table route_legs   add column if not exists typical_duration_min double precision;

comment on column camera_edges.distance_source is
    '''graphhopper'' = real driven road; ''estimate'' = great-circle x 1.25 '
    'fallback used when routing was unavailable.';
comment on column camera_edges.typical_duration_min is
    'GraphHopper''s own driving time for this road. A reference value only - '
    'the acceptance window stays a speed band so it is explainable without a '
    'third party.';
comment on column route_legs.distance_source is
    'Where this leg''s road_distance_km came from; the verdict depends on it.';

-- Surface the provenance wherever legs are read back.
drop view if exists v_route_legs_detail;
create view v_route_legs_detail as
select l.run_id, l.global_vehicle_id, g.plate_number, l.leg_index,
       l.from_camera, l.to_camera, l.departed, l.arrived,
       l.travel_minutes, l.road_distance_km, l.distance_source,
       l.typical_duration_min, l.implied_speed_kmh,
       l.expected_min_minutes, l.expected_max_minutes,
       l.verdict, l.reason
from route_legs l
join global_vehicles g
  on g.run_id = l.run_id and g.global_vehicle_id = l.global_vehicle_id
order by g.plate_number, l.leg_index;
