import { motion } from 'framer-motion'
import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'

/* Formatting helpers. `null` means "the pipeline produced no value", and it is
   rendered as an em dash rather than 0 — a missing measurement and a measured
   zero are different facts. */

export const num = (v: unknown, d = 0): string =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? '—'
    : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })

export const clock = (v?: string | null): string => {
  if (!v) return '—'
  const t = new Date(v)
  return Number.isNaN(t.getTime()) ? String(v) : t.toISOString().replace('T', ' ').slice(0, 19)
}

export const hhmmss = (v?: string | null): string => {
  if (!v) return '—'
  const t = new Date(v)
  return Number.isNaN(t.getTime()) ? String(v) : t.toISOString().slice(11, 19)
}

/* ------------------------------------------------------------ atoms */

export function Panel({
  title, hint, children, flush,
}: { title?: string; hint?: ReactNode; children: ReactNode; flush?: boolean }) {
  return (
    <motion.section
      className="panel"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
    >
      {title && (
        <div className="panel-head">
          <h2>{title}</h2>
          {hint && <span className="hint">{hint}</span>}
        </div>
      )}
      <div className={flush ? 'panel-body flush' : 'panel-body'}>{children}</div>
    </motion.section>
  )
}

/** A number that counts up when it changes — motion that carries meaning. */
export function Counter({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const [shown, setShown] = useState(0)
  const prev = useRef(0)

  useEffect(() => {
    const from = prev.current
    const to = Number(value) || 0
    prev.current = to
    if (from === to) { setShown(to); return }

    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduce) { setShown(to); return }

    const D = 620
    let raf = 0
    const t0 = performance.now()
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / D)
      const eased = 1 - Math.pow(1 - k, 3)
      setShown(from + (to - from) * eased)
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value])

  return <>{num(shown, decimals)}</>
}

export function Tile({
  label, value, sub, accent, decimals = 0, raw,
}: {
  label: string
  value?: number | null
  sub?: ReactNode
  accent?: string
  decimals?: number
  raw?: ReactNode
}) {
  return (
    <motion.div
      className="tile"
      style={{ ['--accent' as string]: accent ?? 'var(--s1)' }}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <div className="label">{label}</div>
      <div className="value" style={accent ? { color: accent } : undefined}>
        {raw ?? (value === null || value === undefined
          ? '—'
          : <Counter value={value} decimals={decimals} />)}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </motion.div>
  )
}

export function Badge({ kind, children }: { kind: string; children: ReactNode }) {
  return <span className={`badge ${kind}`}>{children}</span>
}

/** Says where a number came from. Used everywhere a value could be mistaken
    for a measurement when it is actually inferred or estimated. */
export function Provenance({ real, realLabel = 'real road', estLabel = 'estimate' }:
  { real: boolean; realLabel?: string; estLabel?: string }) {
  return <Badge kind={real ? 'ok' : 'warn'}>{real ? realLabel : estLabel}</Badge>
}

export function verdictKind(v?: string): string {
  switch (v) {
    case 'PLAUSIBLE': return 'ok'
    case 'FAST': return 'warn'
    case 'SLOW': return 'serious'
    case 'IMPLAUSIBLE': return 'critical'
    default: return 'muted'
  }
}

export function Loading({ what = 'Loading…' }: { what?: string }) {
  return (
    <div className="loading">
      <div className="spinner" />
      {what}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  )
}

export const NoRun = (
  <Empty title="No pipeline run in the database">
    Process videos and persist a run first:{' '}
    <code className="mono">python scripts/run_all_cameras.py --save</code>
  </Empty>
)
