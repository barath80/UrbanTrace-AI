-- Make the plate timeline follow the VEHICLE, not the OCR string.
--
-- Association merges OCR variants: TN01AB1234 and TN01A8123 are one vehicle,
-- and the CAM_02 observation is stored under the variant. A timeline that
-- matches observations on plate_number therefore drops that sighting - the
-- tracking view reported "2 cameras" and "1 sighting" for the same vehicle,
-- and drew one marker on a two-stop route.
--
-- The fix is to expose observation_id so the timeline can be resolved through
-- vehicle_observations, which is where the merge is actually recorded.
--
-- Additive and idempotent: safe to run on an existing database.

drop view if exists v_plate_timeline;
create view v_plate_timeline as
select o.run_id, o.observation_id, o.plate_number, o.observed_at, o.video_time_s,
       o.camera_id, c.name as location, c.latitude, c.longitude,
       o.local_track_id, o.ocr_confidence, o.vehicle_type, o.evidence_crop
from observations o
left join cameras c on c.run_id = o.run_id and c.camera_id = o.camera_id
order by o.plate_number, o.observed_at;

-- Every sighting of a vehicle, including sightings recorded under a merged
-- OCR variant of its plate. `plate_number` stays the raw text that was read at
-- that camera; `vehicle_plate` is the identity it was resolved to.
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
