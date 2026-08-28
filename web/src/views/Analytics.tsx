/** Macro traffic analytics: class mix, per-camera counts, detections over time,
    and the origin-destination matrix with measured travel times. */
import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

import {
  getFlows, getTimeline, getVehicleCounts, getVehicleTypes,
  type Flow, type TimelineBucket,
} from '../api'
import { Empty, Loading, Panel, num } from '../components/ui'

const CLASS_COLOR: Record<string, string> = {
  car: '#3987e5', motorcycle: '#d95926', bus: '#199e70',
  truck: '#fab219', bicycle: '#8b6ce6',
}
const CAM_COLORS = ['#3987e5', '#199e70', '#d95926', '#8b6ce6', '#fab219', '#ec835a']

const axis = { stroke: '#7c8899', fontSize: 11 }
const tooltipStyle = {
  background: '#1d2431', border: '1px solid #262f3d', borderRadius: 7, fontSize: 12,
}

export default function Analytics() {
  const [counts, setCounts] = useState<Record<string, unknown>[]>([])
  const [types, setTypes] = useState<Record<string, unknown>[]>([])
  const [timeline, setTimeline] = useState<TimelineBucket[]>([])
  const [flows, setFlows] = useState<Flow[]>([])
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    Promise.all([getVehicleCounts(), getVehicleTypes(), getTimeline(), getFlows()])
      .then(([c, t, tl, f]) => { setCounts(c); setTypes(t); setTimeline(tl); setFlows(f) })
      .catch(() => { /* empty state below */ })
      .finally(() => setBusy(false))
  }, [])

  if (busy) return <Panel><Loading /></Panel>

  const classData = types.map(t => ({
    name: String(t.vehicle_type ?? 'unknown'),
    value: Number(t.observations ?? t.count ?? 0),
  })).filter(d => d.value > 0)

  const countData = counts.map(c => ({
    camera: String(c.camera_id),
    car: Number(c.car ?? 0),
    motorcycle: Number(c.motorcycle ?? 0),
    bus: Number(c.bus ?? 0),
    truck: Number(c.truck ?? 0),
  }))

  const cams = [...new Set(timeline.map(t => t.camera_id))]
  const buckets = [...new Set(timeline.map(t => t.bucket_s))].sort((a, b) => a - b)
  const series = buckets.map(b => {
    const row: Record<string, string | number> = { t: `${b}s` }
    for (const c of cams) {
      row[c] = timeline.find(x => x.bucket_s === b && x.camera_id === c)?.detections ?? 0
    }
    return row
  })

  return (
    <>
      <div className="grid grid-2">
        <Panel title="Unique vehicles by class, per camera"
               hint="deduped tracks, not frame detections">
          {countData.length === 0 ? <Empty title="No counts in this run" /> : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={countData}>
                <CartesianGrid stroke="#1c2431" vertical={false} />
                <XAxis dataKey="camera" {...axis} />
                <YAxis {...axis} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#ffffff08' }} />
                <Legend wrapperStyle={{ fontSize: 11.5 }} />
                {(['car', 'motorcycle', 'bus', 'truck'] as const).map(k => (
                  <Bar key={k} dataKey={k} stackId="a" fill={CLASS_COLOR[k]} radius={0} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel title="Observation mix by vehicle class">
          {classData.length === 0 ? <Empty title="No classified observations" /> : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={classData} dataKey="value" nameKey="name"
                     innerRadius={62} outerRadius={104} paddingAngle={2}>
                  {classData.map(d => (
                    <Cell key={d.name} fill={CLASS_COLOR[d.name] ?? '#7c8899'} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11.5 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <Panel title="Plate reads over time, per camera"
             hint="30-second buckets of measured video time">
        {series.length === 0 ? <Empty title="No time series for this run" /> : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={series}>
              <CartesianGrid stroke="#1c2431" vertical={false} />
              <XAxis dataKey="t" {...axis} />
              <YAxis {...axis} allowDecimals={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11.5 }} />
              {cams.map((c, i) => (
                <Line key={c} type="monotone" dataKey={c} dot={false} strokeWidth={2}
                      stroke={CAM_COLORS[i % CAM_COLORS.length]} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </Panel>

      <Panel title="Origin–destination matrix"
             hint="measured travel time over the real road distance" flush>
        {flows.length === 0 ? (
          <Empty title="No camera-to-camera flows">
            A flow needs the same vehicle identified at two cameras.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>From</th><th>To</th><th className="num">Vehicles</th>
                  <th className="num">Road km</th><th className="num">Mean travel</th>
                  <th className="num">Mean speed</th>
                </tr>
              </thead>
              <tbody>
                {flows.map(f => (
                  <tr key={`${f.from_camera}-${f.to_camera}`}>
                    <td className="mono">{f.from_camera}</td>
                    <td className="mono">{f.to_camera}</td>
                    <td className="num">{f.vehicles}</td>
                    <td className="num">{num(f.road_distance_km, 2)}</td>
                    <td className="num">{num(f.mean_travel_minutes, 1)} min</td>
                    <td className="num">{num(f.mean_speed_kmh, 1)} km/h</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  )
}
