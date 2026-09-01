import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useState } from 'react'

import { getHealth, getSummary, type Health } from './api'
import Alerts from './views/Alerts'
import Analytics from './views/Analytics'
import Cameras from './views/Cameras'
import CityNetwork from './views/CityNetwork'
import CityTrack from './views/CityTrack'
import GisMap from './views/GisMap'
import Live from './views/Live'
import MultiCamera from './views/MultiCamera'
import Observations from './views/Observations'
import Overview from './views/Overview'
import TrackVehicle from './views/TrackVehicle'

type ViewId =
  | 'overview'
  | 'city'
  | 'citytrack'
  | 'multicam'
  | 'live'
  | 'map'
  | 'track'
  | 'analytics'
  | 'cameras'
  | 'alerts'
  | 'data'

const TABS: { id: ViewId; label: string; short: string }[] = [
  { id: 'overview', label: 'Overview', short: 'Home' },
  { id: 'city', label: 'City Analytics', short: 'City' },
  { id: 'citytrack', label: 'Track Vehicle (Network)', short: 'Trace' },
  { id: 'multicam', label: 'Multi-Camera (3-Feed)', short: 'Multi-Cam' },
  { id: 'live', label: 'Live Monitor', short: 'Live' },
  { id: 'map', label: 'GIS Map', short: 'Map' },
  { id: 'track', label: 'Vehicle Tracking', short: 'Track' },
  { id: 'analytics', label: 'Traffic Analytics', short: 'Stats' },
  { id: 'cameras', label: 'Cameras & Video', short: 'Cams' },
  { id: 'alerts', label: 'Alerts', short: 'Alerts' },
  { id: 'data', label: 'Observations', short: 'Data' },
]

function getInitialView(): ViewId {
  const path = window.location.pathname.toLowerCase()
  const hash = window.location.hash.toLowerCase()
  if (path.includes('multi-camera') || hash.includes('multi-camera') || hash.includes('multicam')) {
    return 'multicam'
  }
  return 'overview'
}

export default function App() {
  const [view, setView] = useState<ViewId>(getInitialView)
  const [health, setHealth] = useState<Health | null>(null)
  const [alertCount, setAlertCount] = useState(0)
  const [plate, setPlate] = useState<string | undefined>()

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    getSummary().then(s => setAlertCount(Number(s.alerts) || 0)).catch(() => setAlertCount(0))
  }, [])

  /** Jump straight to a vehicle's dossier from anywhere a plate is shown. */
  const track = useCallback((p: string) => {
    setPlate(p)
    setView('track')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  const rt = health?.runtime

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">UT</div>
          <div>
            <h1>UrbanTrace</h1>
            <p>City-wide ANPR · Trajectory · Traffic Intelligence</p>
          </div>
        </div>
        <div className="chips">
          {/* Database status is not surfaced here: each view reports its own
              empty state, and /api/health still carries the real status for
              anyone who needs it. */}
          <span className="chip hide-sm">
            <span className={`dot ${rt?.cuda ? 'ok' : ''}`} />
            {rt?.gpu_name ?? rt?.device ?? '—'}
          </span>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab${view === t.id ? ' on' : ''}`}
            onClick={() => {
              setView(t.id)
              if (t.id === 'multicam') {
                window.history.pushState(null, '', '/multi-camera')
              } else if (t.id === 'overview') {
                window.history.pushState(null, '', '/')
              }
            }}
          >
            <span className="tab-full">{t.label}</span>
            <span className="tab-short">{t.short}</span>
            {t.id === 'alerts' && alertCount > 0 && (
              <span className="badge-n">{alertCount}</span>
            )}
            {view === t.id && (
              <motion.span className="tab-underline" layoutId="tab-underline"
                           transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
            )}
          </button>
        ))}
      </nav>

      <main>
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            {view === 'overview' && <Overview onTrack={track} />}
            {view === 'city' && <CityNetwork />}
            {view === 'citytrack' && <CityTrack initialPlate={plate} />}
            {view === 'multicam' && <MultiCamera onTrack={track} />}
            {view === 'live' && <Live onTrack={track} />}
            {view === 'map' && <GisMap onTrack={track} />}
            {view === 'track' && <TrackVehicle initialPlate={plate} />}
            {view === 'analytics' && <Analytics />}
            {view === 'cameras' && <Cameras />}
            {view === 'alerts' && <Alerts onTrack={track} />}
            {view === 'data' && <Observations onTrack={track} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
