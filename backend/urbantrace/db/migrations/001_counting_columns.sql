-- Counting columns for camera_stats.
--
-- Unique-vehicle counting moved off "one ByteTrack id = one vehicle" (which
-- overcounts: fragmented tracks, kerbside clutter, few-frame flicker) onto
-- vision/counting.py, which stitches fragments and gates out clutter. That
-- makes the stored count ambiguous unless the row also records WHICH count it
-- is, so count_mode is now part of the record.
--
-- Additive and idempotent: safe to run on an existing database.

alter table camera_stats add column if not exists count_mode               text;
alter table camera_stats add column if not exists count_traffic            int;
alter table camera_stats add column if not exists count_presence           int;
alter table camera_stats add column if not exists count_fragments_stitched int;
alter table camera_stats add column if not exists count_rejected           int;

comment on column camera_stats.count_mode is
    'Which count unique_vehicles is: ''traffic'' (vehicles that travelled '
    'through the scene, or crossed the camera''s counting line) or '
    '''presence'' (every distinct vehicle that appeared, parked included).';
comment on column camera_stats.count_fragments_stitched is
    'Tracker-id fragments rejoined into an existing vehicle. High values mean '
    'the tracker is losing vehicles often - a tracking-quality signal.';
comment on column camera_stats.count_rejected is
    'Tracks rejected as clutter or flicker before counting.';

-- Dropped rather than replaced: the view gains columns in the middle of its
-- list, and CREATE OR REPLACE VIEW cannot rename or reorder existing columns.
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
