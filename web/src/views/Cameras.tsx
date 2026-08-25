/** Camera nodes with their annotated output video and measured throughput. */
import { useEffect, useState } from 'react'

import { getCameras, type Camera } from '../api'
import { Badge, Empty, Loading, NoRun, Panel, num } from '../components/ui'

export default function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [busy, setBusy] = useState(true)
  const [err, setErr] = useState(false)

  useEffect(() => {
    getCameras().then(setCameras).catch(() => setErr(true)).finally(() => setBusy(false))
  }, [])

  if (busy) return <Panel><Loading /></Panel>
  if (err) return <Panel>{NoRun}</Panel>
  if (!cameras.length) return <Panel><Empty title="No cameras in this run" /></Panel>

  return (
    <div className="grid grid-2">
      {cameras.map(c => (
        <div className="video-card" key={c.camera_id}>
          <div className="video-head">
            <h3>
              {c.camera_id} — {c.name}{' '}
              <Badge kind={c.location_source === 'config' ? 'ok' : 'muted'}>
                {c.location_source}
              </Badge>
            </h3>
            <p>
              {c.road} · {num(c.latitude, 4)}, {num(c.longitude, 4)} · {c.direction}
            </p>
          </div>
          {c.has_video ? (
            <video controls preload="metadata" src={`/api/video/${c.camera_id}`} />
          ) : (
            <Empty title="No annotated video on disk">
              Run the pipeline without <code className="mono">--no-video</code>.
            </Empty>
          )}
          <div className="video-foot">
            <span><b>{c.unique_vehicles}</b> unique vehicles ({c.count_mode ?? 'traffic'})</span>
            <span><b>{c.count_car}</b> car</span>
            <span><b>{c.count_motorcycle}</b> moto</span>
            <span><b>{c.count_bus}</b> bus</span>
            <span><b>{c.count_truck}</b> truck</span>
            <span><b>{c.vehicles_with_plate}</b> identified</span>
            <span>{num(c.processing_fps, 1)} fps processed</span>
          </div>
        </div>
      ))}
    </div>
  )
}
