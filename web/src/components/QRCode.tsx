/**
 * A QR code, drawn as an SVG, with no external dependency and no network call.
 *
 * The phone setup addresses are LAN addresses (http://192.168.x.x:8000/...),
 * so a hosted QR image service would leak them and would be useless anyway
 * when the laptop is running as an offline hotspot. This encodes locally.
 *
 * Scope: byte mode, error-correction level M, versions 1 to 9 (up to 180
 * characters). That covers every address this app shows; longer input is
 * reported rather than rendered as a broken code.
 */

/* ---- Galois field GF(256) for Reed-Solomon ------------------------- */

const EXP = new Uint8Array(512)
const LOG = new Uint8Array(256)
for (let i = 0, x = 1; i < 255; i++) {
  EXP[i] = x
  LOG[x] = i
  x <<= 1
  if (x & 0x100) x ^= 0x11d
}
for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255]

const mul = (a: number, b: number) => (a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]])

/** Generator polynomial for `degree` error-correction codewords. */
function generator(degree: number): Uint8Array {
  let poly = new Uint8Array([1])
  for (let d = 0; d < degree; d++) {
    const next = new Uint8Array(poly.length + 1)
    for (let i = 0; i < poly.length; i++) {
      next[i] ^= poly[i]
      next[i + 1] ^= mul(poly[i], EXP[d])
    }
    poly = next
  }
  return poly
}

/** The `degree` error-correction codewords for one data block. */
function ecCodewords(data: Uint8Array, degree: number): Uint8Array {
  const gen = generator(degree)
  const rem = new Uint8Array(degree)
  for (const byte of data) {
    const factor = byte ^ rem[0]
    rem.copyWithin(0, 1)
    rem[degree - 1] = 0
    if (factor !== 0) {
      for (let i = 0; i < degree; i++) rem[i] ^= mul(gen[i + 1], factor)
    }
  }
  return rem
}

/* ---- version tables (error-correction level M, versions 1-9) -------- */

/** [ec codewords per block, blocks in group 1, data per block, blocks in group 2, data per block] */
const BLOCKS: Record<number, [number, number, number, number, number]> = {
  1: [10, 1, 16, 0, 0],
  2: [16, 1, 28, 0, 0],
  3: [26, 1, 44, 0, 0],
  4: [18, 2, 32, 0, 0],
  5: [24, 2, 43, 0, 0],
  6: [16, 4, 27, 0, 0],
  7: [18, 4, 31, 0, 0],
  8: [22, 2, 38, 2, 39],
  9: [22, 3, 36, 2, 37],
}

/** Alignment pattern centre coordinates, by version. */
const ALIGN: Record<number, number[]> = {
  1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
  6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46],
}

/** Total data codewords available at a version. */
function dataCapacity(version: number): number {
  const [, n1, d1, n2, d2] = BLOCKS[version]
  return n1 * d1 + n2 * d2
}

/* ---- bit stream ---------------------------------------------------- */

class Bits {
  readonly bits: number[] = []
  push(value: number, length: number) {
    for (let i = length - 1; i >= 0; i--) this.bits.push((value >>> i) & 1)
  }
  get length() { return this.bits.length }
  toBytes(): Uint8Array {
    const out = new Uint8Array(Math.ceil(this.bits.length / 8))
    this.bits.forEach((b, i) => { if (b) out[i >> 3] |= 0x80 >>> (i & 7) })
    return out
  }
}

/* ---- mask patterns and penalties ----------------------------------- */

const MASKS: ((r: number, c: number) => boolean)[] = [
  (r, c) => (r + c) % 2 === 0,
  r => r % 2 === 0,
  (_r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
]

/** The four penalty rules from the specification, summed. */
function penalty(m: boolean[][]): number {
  const n = m.length
  let score = 0

  // Rule 1: runs of five or more same-coloured modules in a row or column.
  for (let i = 0; i < n; i++) {
    for (const read of [(k: number) => m[i][k], (k: number) => m[k][i]]) {
      let run = 1
      for (let k = 1; k < n; k++) {
        if (read(k) === read(k - 1)) run++
        else { if (run >= 5) score += run - 2; run = 1 }
      }
      if (run >= 5) score += run - 2
    }
  }

  // Rule 2: 2x2 blocks of a single colour.
  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const v = m[r][c]
      if (v === m[r][c + 1] && v === m[r + 1][c] && v === m[r + 1][c + 1]) score += 3
    }
  }

  // Rule 3: finder-like 1:1:3:1:1 patterns with four light modules beside them.
  const a = [true, false, true, true, true, false, true, false, false, false, false]
  const b = [false, false, false, false, true, false, true, true, true, false, true]
  const matches = (read: (k: number) => boolean, start: number, pat: boolean[]) => {
    for (let k = 0; k < pat.length; k++) if (read(start + k) !== pat[k]) return false
    return true
  }
  for (let i = 0; i < n; i++) {
    for (const read of [(k: number) => m[i][k], (k: number) => m[k][i]]) {
      for (let s = 0; s + 11 <= n; s++) {
        if (matches(read, s, a) || matches(read, s, b)) score += 40
      }
    }
  }

  // Rule 4: deviation from an even balance of dark and light modules.
  let dark = 0
  for (const row of m) for (const v of row) if (v) dark++
  score += Math.floor(Math.abs((dark * 100) / (n * n) - 50) / 5) * 10

  return score
}

/* ---- matrix construction ------------------------------------------- */

type Grid = (boolean | null)[][]

/** Finders, separators, timing, alignment, and the reserved format areas. */
function placeFunctionPatterns(m: Grid, version: number) {
  const n = m.length

  const finder = (top: number, left: number) => {
    for (let r = -1; r <= 7; r++) {
      for (let c = -1; c <= 7; c++) {
        const rr = top + r, cc = left + c
        if (rr < 0 || rr >= n || cc < 0 || cc >= n) continue
        const edge = r === 0 || r === 6 || c === 0 || c === 6
        const core = r >= 2 && r <= 4 && c >= 2 && c <= 4
        m[rr][cc] = (edge || core) && r >= 0 && r <= 6 && c >= 0 && c <= 6
      }
    }
  }
  finder(0, 0)
  finder(0, n - 7)
  finder(n - 7, 0)

  for (let i = 8; i < n - 8; i++) {
    m[6][i] = i % 2 === 0
    m[i][6] = i % 2 === 0
  }

  for (const r of ALIGN[version]) {
    for (const c of ALIGN[version]) {
      if ((r === 6 && c === 6) || (r === 6 && c === n - 7) || (r === n - 7 && c === 6)) continue
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          m[r + dr][c + dc] = Math.max(Math.abs(dr), Math.abs(dc)) !== 1
        }
      }
    }
  }

  // Version information, present from version 7 up: an 18-bit BCH word
  // repeated beside the bottom-left and top-right finders.
  if (version >= 7) {
    let rem = version
    for (let i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >>> 11) * 0x1f25)
    const word = (version << 12) | rem
    for (let i = 0; i < 18; i++) {
      const bit = ((word >>> i) & 1) === 1
      const a = n - 11 + (i % 3)
      const b = Math.floor(i / 3)
      m[b][a] = bit
      m[a][b] = bit
    }
  }

  // Reserve the format-information areas so data placement skips them.
  for (let i = 0; i < 9; i++) {
    if (m[8][i] === null) m[8][i] = false
    if (m[i][8] === null) m[i][8] = false
  }
  for (let i = 0; i < 8; i++) {
    if (m[8][n - 1 - i] === null) m[8][n - 1 - i] = false
    if (m[n - 1 - i][8] === null) m[n - 1 - i][8] = false
  }
}

/** 15-bit BCH format information for level M and the given mask. */
function formatBits(mask: number): number {
  const data = (0b00 << 3) | mask // 00 = error-correction level M
  let rem = data << 10
  for (let i = 14; i >= 10; i--) {
    if ((rem >>> i) & 1) rem ^= 0b10100110111 << (i - 10)
  }
  return ((data << 10) | (rem & 0x3ff)) ^ 0b101010000010010
}

function placeFormat(m: boolean[][], mask: number) {
  const n = m.length
  const bits = formatBits(mask)
  const at = (i: number) => ((bits >>> i) & 1) === 1

  // First copy: down the right side of the top-left finder, then along its
  // underside.
  for (let i = 0; i <= 5; i++) m[i][8] = at(i)
  m[7][8] = at(6)
  m[8][8] = at(7)
  m[8][7] = at(8)
  for (let i = 9; i <= 14; i++) m[8][14 - i] = at(i)

  // Second copy: beside the bottom-left finder, then above the top-right one.
  // This copy runs in the opposite bit order to the first.
  for (let i = 0; i < 7; i++) m[n - 1 - i][8] = at(14 - i)
  for (let i = 0; i < 8; i++) m[8][n - 8 + i] = at(7 - i)

  m[n - 8][8] = true // the always-dark module, which overlays the copy above
}

/** Interleave the data and error-correction blocks into the final codeword stream. */
function interleave(data: Uint8Array, version: number): Uint8Array {
  const [ecLen, n1, d1, n2, d2] = BLOCKS[version]
  const dataBlocks: Uint8Array[] = []
  const ecBlocks: Uint8Array[] = []

  let offset = 0
  for (const [count, size] of [[n1, d1], [n2, d2]]) {
    for (let b = 0; b < count; b++) {
      const block = data.subarray(offset, offset + size)
      offset += size
      dataBlocks.push(block)
      ecBlocks.push(ecCodewords(block, ecLen))
    }
  }

  const out: number[] = []
  for (let i = 0; i < Math.max(d1, d2); i++) {
    for (const block of dataBlocks) if (i < block.length) out.push(block[i])
  }
  for (let i = 0; i < ecLen; i++) {
    for (const block of ecBlocks) out.push(block[i])
  }
  return Uint8Array.from(out)
}

/** Encode `text` and return the module matrix (true = dark), without a quiet zone. */
function encodeQR(text: string): boolean[][] {
  const bytes = new TextEncoder().encode(text)

  let version = 0
  for (let v = 1; v <= 9; v++) {
    // 4 mode bits + 8 character-count bits + the payload.
    if (12 + bytes.length * 8 <= dataCapacity(v) * 8) { version = v; break }
  }
  if (!version) throw new Error(`QR payload too long (${bytes.length} bytes, limit 180)`)

  const capacity = dataCapacity(version)
  const bits = new Bits()
  bits.push(0b0100, 4)       // byte mode
  bits.push(bytes.length, 8) // character count, versions 1-9
  for (const b of bytes) bits.push(b, 8)
  bits.push(0, Math.min(4, capacity * 8 - bits.length)) // terminator
  while (bits.length % 8 !== 0) bits.push(0, 1)

  const used = bits.length / 8
  const codewords = new Uint8Array(capacity)
  codewords.set(bits.toBytes())
  for (let i = used; i < capacity; i++) codewords[i] = (i - used) % 2 === 0 ? 0xec : 0x11

  const final = interleave(codewords, version)
  const n = version * 4 + 17

  // Build one candidate per mask and keep the least penalised, as the spec requires.
  let best: boolean[][] | null = null
  let bestScore = Infinity
  for (let mask = 0; mask < 8; mask++) {
    const grid: Grid = Array.from({ length: n }, () => Array<boolean | null>(n).fill(null))
    placeFunctionPatterns(grid, version)

    let bit = 0
    let up = true
    let col = n - 1
    while (col > 0) {
      if (col === 6) col = 5 // column 6 is the vertical timing pattern
      for (let i = 0; i < n; i++) {
        const row = up ? n - 1 - i : i
        for (const c of [col, col - 1]) {
          if (grid[row][c] !== null) continue
          let dark = bit < final.length * 8 && ((final[bit >> 3] >>> (7 - (bit & 7))) & 1) === 1
          bit++
          if (MASKS[mask](row, c)) dark = !dark
          grid[row][c] = dark
        }
      }
      up = !up
      col -= 2
    }

    const solid = grid.map(row => row.map(v => v === true))
    placeFormat(solid, mask)
    const score = penalty(solid)
    if (score < bestScore) { bestScore = score; best = solid }
  }

  return best!
}

/* ---- component ----------------------------------------------------- */

export default function QRCode({
  value,
  size = 132,
  label,
  className,
}: {
  value: string
  /** Rendered width in pixels; the code is square. */
  size?: number
  /** Accessible description of what the code points at. */
  label?: string
  className?: string
}) {
  let modules: boolean[][]
  try {
    modules = encodeQR(value)
  } catch {
    return <div className={`qr-error${className ? ` ${className}` : ''}`}>QR unavailable</div>
  }

  const quiet = 4 // the light margin a scanner needs around the code
  const dim = modules.length + quiet * 2

  const path = modules
    .flatMap((row, r) => row.map((dark, c) => (dark ? `M${c + quiet} ${r + quiet}h1v1h-1z` : '')))
    .join('')

  return (
    <svg
      className={`qr${className ? ` ${className}` : ''}`}
      width={size}
      height={size}
      viewBox={`0 0 ${dim} ${dim}`}
      shapeRendering="crispEdges"
      role="img"
      aria-label={label ?? `QR code for ${value}`}
    >
      <rect width={dim} height={dim} fill="#ffffff" />
      <path d={path} fill="#000000" />
    </svg>
  )
}
