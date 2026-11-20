import { ChangeEvent, CSSProperties, DragEvent, FormEvent, useEffect, useMemo, useState } from 'react'
import { AlertCircle, ArrowRight, Check, ImagePlus, LoaderCircle, RotateCcw, Ruler, Upload, X } from 'lucide-react'
import { Magnetic, useSpring } from './fx'
import { MaskedLines, Reveal } from './reveal'

export type ModelStatus = 'checking' | 'ready' | 'unavailable'
type Mode = 'specs' | 'photo'
type NumericKey = 'width_cm' | 'depth_cm' | 'height_cm' | 'price_kzt' | 'weight_kg'
type FormState = Record<NumericKey, string> & { materials: string[]; main_color: string }
type Probability = { type: string; probability: number }
type Prediction = { prediction: string; confidence: number; probabilities: Probability[]; source: string; model: string; cv_macro_f1: number }

/* Row-major positions in the generated 4 × 4 material texture atlas. */
const MATERIALS = [
  'chipboard', 'mdf', 'fibreboard', 'solid_wood',
  'plywood', 'veneer', 'metal', 'plastic',
  'glass', 'mirror', 'fabric', 'leather',
  'foam', 'spring', 'rattan', 'stone',
]
// Flat, matte, saturated paint chips.
const COLOURS: Record<string, string> = {
  white: '#f2efe9', black: '#141414', grey: '#7a7e84', beige: '#e3c597', brown: '#7c4119', green: '#2e8b4a',
  blue: '#1d5fc4', light_blue: '#4fb3ec', pink: '#ee7aa5', red: '#d42f27', yellow: '#f4c22b', orange: '#f27716',
  purple: '#7b3cb4', burgundy: '#8c1430', terracotta: '#d1572a', olive: '#7f8b24', mint: '#4fd4a4',
  other: 'conic-gradient(#d42f27, #f4c22b, #2e8b4a, #1d5fc4, #7b3cb4, #d42f27)',
}
/** Same limits as `api.py` (validate_number). */
const LIMITS: Record<NumericKey, { min: number; max: number; slideMax: number; label: string; unit: string; log?: boolean; step: number }> = {
  width_cm: { min: 10, max: 500, slideMax: 320, label: 'Width', unit: 'cm', step: 1 },
  depth_cm: { min: 10, max: 500, slideMax: 320, label: 'Depth', unit: 'cm', step: 1 },
  height_cm: { min: 10, max: 500, slideMax: 320, label: 'Height', unit: 'cm', step: 1 },
  price_kzt: { min: 5_000, max: 10_000_000, slideMax: 10_000_000, label: 'Listed price', unit: 'KZT', log: true, step: 1000 },
  weight_kg: { min: 0.5, max: 500, slideMax: 500, label: 'Weight', unit: 'kg', log: true, step: 0.5 },
}
/** Values copied from typical real mebel.kz listings of each type. */
const EXAMPLES: { name: string; values: Partial<FormState> }[] = [
  { name: 'Chest of drawers', values: { width_cm: '105', depth_cm: '42', height_cm: '80', price_kzt: '114770', weight_kg: '40', materials: ['chipboard', 'solid_wood'], main_color: 'white' } },
  { name: 'Sofa', values: { width_cm: '220', depth_cm: '94', height_cm: '93', price_kzt: '421010', weight_kg: '70', materials: ['chipboard', 'fibreboard', 'plywood', 'solid_wood', 'spring'], main_color: 'grey' } },
  { name: 'Wardrobe', values: { width_cm: '160', depth_cm: '60', height_cm: '220', price_kzt: '280000', weight_kg: '110', materials: ['chipboard', 'mirror'], main_color: 'beige' } },
]
const INITIAL: FormState = { width_cm: '120', depth_cm: '45', height_cm: '75', price_kzt: '150000', weight_kg: '40', materials: ['chipboard'], main_color: 'white' }
const LABELS: Record<string, string> = { solid_wood: 'Solid wood', light_blue: 'Light blue', tv_stand: 'TV stand', mdf: 'MDF', chest_of_drawers: 'Chest of drawers', coffee_table: 'Coffee table' }
const MAX_PHOTO = 10 * 1024 * 1024

export const pretty = (value: string) => LABELS[value] ?? value.replaceAll('_', ' ').replace(/^\w/, (l) => l.toUpperCase())

function checkNumber(key: NumericKey, raw: string) {
  const { min, max, label } = LIMITS[key]
  if (raw.trim() === '') return `${label} is required.`
  const value = Number(raw)
  if (!Number.isFinite(value)) return `${label} must be a number.`
  if (value < min || value > max) return `${label} must be between ${min.toLocaleString('en')} and ${max.toLocaleString('en')}.`
  return ''
}

/* ---------- Ruler slider: big editable number + tick-marked track ---------- */
function RulerField({ field, value, error, onChange }: { field: NumericKey; value: string; error: string; onChange: (k: NumericKey, v: string) => void }) {
  const { min, max, slideMax, label, unit, log, step } = LIMITS[field]
  const top = Math.min(max, slideMax)
  const num = Number(value)
  const valid = value.trim() !== '' && Number.isFinite(num)
  const toT = (v: number) => {
    const c = Math.min(Math.max(v, min), top)
    return log ? Math.log(c / min) / Math.log(top / min) : (c - min) / (top - min)
  }
  const fromT = (t: number) => {
    const raw = log ? min * Math.pow(top / min, t) : min + t * (top - min)
    const snap = log && raw > 10_000 ? 1000 : step
    return String(Math.round(raw / snap) * snap)
  }
  const t = valid ? toT(num) : 0
  return (
    <div className={`ruler ${error ? 'has-error' : ''}`} style={{ '--t': t } as CSSProperties}>
      <label className="ruler-head">
        <span className="ruler-label">{label}</span>
        <span className="ruler-value">
          <input
            type="number" inputMode="decimal" name={field} value={value} min={min} max={max} step={step}
            aria-invalid={Boolean(error)} aria-describedby={error ? `${field}-error` : undefined}
            onChange={(event) => onChange(field, event.target.value)}
            style={{ width: `${Math.max(2, value.length) + 0.6}ch` }}
          />
          <em>{unit}</em>
        </span>
      </label>
      <input
        className="ruler-track" type="range" min={0} max={1000} step={1} value={Math.round(t * 1000)}
        aria-label={`${label} slider`} onChange={(event) => onChange(field, fromT(Number(event.target.value) / 1000))}
      />
      {error && <span className="field-error" id={`${field}-error`}>{error}</span>}
    </div>
  )
}

/* ---------- Live isometric outline that morphs with the sizes ---------- */
function LiveOutline({ w, d, h, busy }: { w: number; d: number; h: number; busy: boolean }) {
  const safe = (v: number) => (Number.isFinite(v) && v > 0 ? Math.min(v, 500) : 10)
  const W0 = useSpring(safe(w)), D0 = useSpring(safe(d)), H0 = useSpring(safe(h))
  const scale = 230 / Math.max(W0 + D0 * 0.62, (H0 + D0 * 0.36) * 1.25, 1)
  const W = W0 * scale, D = D0 * scale * 0.62, H = H0 * scale, rise = D * 0.58
  const x0 = 170 - (W + D) / 2, y0 = 170 + (H - rise) / 2 + 6
  const p = (x: number, y: number) => `${(x0 + x).toFixed(1)},${(y0 + y).toFixed(1)}`
  const front = [p(0, 0), p(W, 0), p(W, -H), p(0, -H)].join(' ')
  const top = [p(0, -H), p(W, -H), p(W + D, -H - rise), p(D, -H - rise)].join(' ')
  const side = [p(W, 0), p(W + D, -rise), p(W + D, -H - rise), p(W, -H)].join(' ')
  return (
    <svg className={`live-outline ${busy ? 'is-busy' : ''}`} viewBox="0 0 340 340" role="img" aria-label={`Outline ${Math.round(w)} by ${Math.round(d)} by ${Math.round(h)} centimetres`}>
      <defs>
        <linearGradient id="lo-front" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--honey)" stopOpacity=".26" /><stop offset="1" stopColor="var(--honey)" stopOpacity=".06" /></linearGradient>
        <radialGradient id="lo-floor"><stop offset="0" stopColor="#000" stopOpacity=".45" /><stop offset="1" stopColor="#000" stopOpacity="0" /></radialGradient>
      </defs>
      <ellipse cx={x0 + (W + D) / 2} cy={y0 + 8} rx={(W + D) * 0.62} ry={14 + D * 0.2} fill="url(#lo-floor)" />
      <polygon points={top} className="lo-face lo-top" />
      <polygon points={side} className="lo-face lo-side" />
      <polygon points={front} className="lo-face lo-front" fill="url(#lo-front)" />
      <g className="lo-dims">
        <line x1={x0} y1={y0 + 18} x2={x0 + W} y2={y0 + 18} />
        <text x={x0 + W / 2} y={y0 + 34} textAnchor="middle">{Math.round(w)} W</text>
        <line x1={x0 - 14} y1={y0} x2={x0 - 14} y2={y0 - H} />
        <text x={x0 - 22} y={y0 - H / 2} textAnchor="end" dominantBaseline="middle">{Math.round(h)} H</text>
        <text x={x0 + W + D + 8} y={y0 - rise - H / 2} dominantBaseline="middle">{Math.round(d)} D</text>
      </g>
    </svg>
  )
}

/* ---------- Result ---------- */
function ResultView({ result }: { result: Prediction }) {
  const C = 2 * Math.PI * 44
  const name = pretty(result.prediction)
  const uncertain = result.confidence < 0.6
  return (
    <div className="result" aria-live="polite">
      <div className="result-top">
        <div>
          <p className="eyebrow light">It looks like a</p>
          <h2 className="result-name" aria-label={name}>
            {[...name].map((ch, i) => <span key={i} style={{ '--i': i } as CSSProperties} aria-hidden="true">{ch === ' ' ? ' ' : ch}</span>)}
          </h2>
        </div>
        <div className="ring" style={{ '--c': C, '--off': C * (1 - result.confidence) } as CSSProperties}>
          <svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="44" className="ring-bg" /><circle cx="50" cy="50" r="44" className="ring-fg" /></svg>
          <span><strong>{Math.round(result.confidence * 100)}</strong><small>%</small></span>
        </div>
      </div>
      {uncertain && <p className="note">The top types are close. Read this as a suggestion, not an answer.</p>}
      <ol className="odds">
        {result.probabilities.slice(0, 5).map((item, index) => (
          <li key={item.type} style={{ '--i': index, '--w': `${Math.max(item.probability * 100, 0.8)}%` } as CSSProperties}>
            <span className="odds-name">{pretty(item.type)}</span>
            <span className="odds-bar"><i /></span>
            <span className="odds-num">{(item.probability * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
      <dl className="facts">
        <div><dt>Model</dt><dd>{result.model}</dd></div>
        <div><dt>Grouped CV macro-F1</dt><dd>{result.cv_macro_f1.toFixed(3)}</dd></div>
        <div><dt>Input</dt><dd>{result.source}</dd></div>
      </dl>
      <p className="fine">The percentage is the model's own estimate over nine furniture types, not a guarantee. The macro-F1 shows how often this model was right on listings it never saw.</p>
    </div>
  )
}

/* ---------- Page ---------- */
export default function PredictPage({ status }: { status: ModelStatus }) {
  const [backgroundReady, setBackgroundReady] = useState(false)
  const [mode, setMode] = useState<Mode>('specs')
  const [form, setForm] = useState<FormState>(INITIAL)
  const [touched, setTouched] = useState(false)
  const [useSpecs, setUseSpecs] = useState(false)
  const [photo, setPhoto] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const preview = useMemo(() => (photo ? URL.createObjectURL(photo) : ''), [photo])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const needsSpecs = mode === 'specs' || useSpecs
  const fields: NumericKey[] = mode === 'specs' ? ['width_cm', 'depth_cm', 'height_cm', 'price_kzt'] : ['width_cm', 'depth_cm', 'height_cm', 'weight_kg']
  const errors = Object.fromEntries(fields.map((k) => [k, checkNumber(k, form[k])])) as Record<NumericKey, string>
  const specsValid = !needsSpecs || fields.every((k) => !errors[k])

  const clear = () => { setResult(null); setError('') }
  const changeMode = (next: Mode) => { setMode(next); setTouched(false); clear() }
  const setNumber = (k: NumericKey, v: string) => { setForm((c) => ({ ...c, [k]: v })); setTouched(true); setResult(null) }
  const toggleMaterial = (m: string) => { setForm((c) => ({ ...c, materials: c.materials.includes(m) ? c.materials.filter((x) => x !== m) : [...c.materials, m] })); setResult(null) }
  const applyExample = (values: Partial<FormState>) => { setForm((c) => ({ ...c, ...values })); setTouched(false); clear() }
  const resetAll = () => { setForm(INITIAL); setPhoto(null); setUseSpecs(false); setTouched(false); clear() }

  function takeFile(next: File | null) {
    clear()
    if (!next) return
    if (!['image/jpeg', 'image/png'].includes(next.type)) { setError('The photo must be a JPG or PNG file.'); return }
    if (next.size > MAX_PHOTO) { setError('Choose a photo smaller than 10 MB.'); return }
    setPhoto(next)
  }
  const choosePhoto = (e: ChangeEvent<HTMLInputElement>) => { takeFile(e.target.files?.[0] ?? null); e.target.value = '' }
  const dropPhoto = (e: DragEvent<HTMLLabelElement>) => { e.preventDefault(); setDragging(false); takeFile(e.dataTransfer.files?.[0] ?? null) }

  async function submit(event: FormEvent) {
    event.preventDefault()
    setTouched(true)
    if (!specsValid) { setError('Fix the highlighted fields first.'); return }
    if (mode === 'photo' && !photo) { setError('Choose a furniture photo before running the model.'); return }
    setLoading(true); setError(''); setResult(null)
    try {
      let response: Response
      if (mode === 'specs') {
        response = await fetch('/api/predict/specs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            width_cm: Number(form.width_cm), depth_cm: Number(form.depth_cm), height_cm: Number(form.height_cm),
            price_kzt: Number(form.price_kzt), materials: form.materials, main_color: form.main_color,
          }),
        })
      } else {
        const payload = new FormData()
        payload.append('photo', photo as File)
        payload.append('use_specs', String(useSpecs))
        if (useSpecs) {
          ;(['width_cm', 'depth_cm', 'height_cm', 'weight_kg'] as NumericKey[]).forEach((k) => payload.append(k, form[k]))
          payload.append('main_color', form.main_color)
          form.materials.forEach((m) => payload.append('materials', m))
        }
        response = await fetch('/api/predict/photo', { method: 'POST', body: payload })
      }
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || `The model server answered with status ${response.status}.`)
      setResult(body)
    } catch (problem) {
      setError(problem instanceof TypeError
        ? 'The model server is not reachable. Start it with `python api.py` and reload this page.'
        : problem instanceof Error ? problem.message : 'The model could not complete this prediction.')
    } finally {
      setLoading(false)
    }
  }

  let step = 0
  const num = () => String(++step).padStart(2, '0')

  return (
    <div className="pp">
      <div className="pp-media" aria-hidden="true">
        <div className="pp-media-frame">
          <img src="/hero-minimal.jpg" alt="" />
          <video className={`pp-video${backgroundReady ? ' is-ready' : ''}`} autoPlay muted loop playsInline preload="metadata" onCanPlay={() => setBackgroundReady(true)} onError={() => setBackgroundReady(false)}>
            <source src="/hero-background.mp4" type="video/mp4" />
          </video>
        </div>
      </div>
      <div className="pp-shade" aria-hidden="true" />

      <header className="pp-head">
        <Reveal delay={60}><p className="eyebrow light">Predict</p></Reveal>
        <MaskedLines as="h1" lines={['What kind of', 'furniture is it?']} className="pp-title" start={120} immediate />
        <Reveal delay={420}>
          <div className="modes" role="tablist" aria-label="Prediction input">
            <button type="button" role="tab" aria-selected={mode === 'specs'} className={mode === 'specs' ? 'on' : ''} onClick={() => changeMode('specs')}><Ruler size={16} /> From specifications</button>
            <button type="button" role="tab" aria-selected={mode === 'photo'} className={mode === 'photo' ? 'on' : ''} onClick={() => changeMode('photo')}><ImagePlus size={16} /> From a photo</button>
            <span className="modes-thumb" data-pos={mode} aria-hidden="true" />
          </div>
        </Reveal>
        {status === 'unavailable' && (
          <p className="offline"><AlertCircle size={15} /> The model server is offline. Start it with <code>python api.py</code>, then reload.</p>
        )}
      </header>

      <form className="workbench" onSubmit={submit} noValidate>
        <Reveal delay={520} className="bench" key={mode}>
          {mode === 'specs' && (
            <div className="presets">
              <span className="eyebrow light">Load a real listing</span>
              <div>{EXAMPLES.map((ex) => <button type="button" key={ex.name} onClick={() => applyExample(ex.values)}>{ex.name}</button>)}</div>
            </div>
          )}

          {mode === 'photo' && (
            <section className="step">
              <header><span className="step-no">{num()}</span><h3>Photo</h3><p>One clear JPG or PNG, whole piece in frame, plain background works best.</p></header>
              {photo ? (
                <div className="photo-chip">
                  <img src={preview} alt="" />
                  <span>{photo.name}</span>
                  <button type="button" onClick={() => { setPhoto(null); clear() }} aria-label="Remove photo"><X size={16} /></button>
                </div>
              ) : (
                <label className={`dropzone ${dragging ? 'is-drag' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={dropPhoto}>
                  <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
                  <Upload size={22} />
                  <strong>Drop a photo here</strong>
                  <span>or click to browse · JPG, PNG · up to 10 MB</span>
                  <input type="file" accept="image/jpeg,image/png" onChange={choosePhoto} />
                </label>
              )}
              <label className="switch-row">
                <span><strong>Add specifications</strong><small>Photo + sizes scored 0.794 vs 0.771 for the photo alone.</small></span>
                <input type="checkbox" checked={useSpecs} onChange={(e) => { setUseSpecs(e.target.checked); clear() }} />
                <i aria-hidden="true" />
              </label>
            </section>
          )}

          {needsSpecs && (
            <>
              <section className="step">
                <header><span className="step-no">{num()}</span><h3>Size</h3><p>Drag the rulers or type the numbers from the product page.</p></header>
                <div className="rulers">
                  {(['width_cm', 'depth_cm', 'height_cm'] as NumericKey[]).map((k) => (
                    <RulerField key={k} field={k} value={form[k]} error={touched ? errors[k] : ''} onChange={setNumber} />
                  ))}
                </div>
              </section>

              <section className="step">
                <header><span className="step-no">{num()}</span><h3>{mode === 'specs' ? 'Price' : 'Weight'}</h3><p>{mode === 'specs' ? 'The listed price in tenge.' : 'Shipping weight in kilograms.'}</p></header>
                {mode === 'specs'
                  ? <RulerField field="price_kzt" value={form.price_kzt} error={touched ? errors.price_kzt : ''} onChange={setNumber} />
                  : <RulerField field="weight_kg" value={form.weight_kg} error={touched ? errors.weight_kg : ''} onChange={setNumber} />}
              </section>

              <section className="step">
                <header><span className="step-no">{num()}</span><h3>Materials</h3><p>{form.materials.length ? `${form.materials.length} selected` : 'None selected: the model will treat every material as absent.'}</p></header>
                <div className="swatches">
                  {MATERIALS.map((key, index) => {
                    const on = form.materials.includes(key)
                    return (
                      <button type="button" key={key} className={on ? 'on' : ''} aria-pressed={on} onClick={() => toggleMaterial(key)}>
                        <i aria-hidden="true" style={{ backgroundPosition: `${(index % 4) * 100 / 3}% ${Math.floor(index / 4) * 100 / 3}%` }}>{on && <Check size={13} strokeWidth={3} />}</i>
                        <span>{pretty(key)}</span>
                      </button>
                    )
                  })}
                </div>
              </section>

              <section className="step">
                <header><span className="step-no">{num()}</span><h3>Main colour</h3><p>{pretty(form.main_color)}</p></header>
                <div className="dots" role="radiogroup" aria-label="Main colour">
                  {Object.entries(COLOURS).map(([key, paint]) => (
                    <button type="button" key={key} role="radio" aria-checked={form.main_color === key} aria-label={pretty(key)} title={pretty(key)}
                      className={form.main_color === key ? 'on' : ''} style={{ '--paint': paint } as CSSProperties}
                      onClick={() => { setForm((c) => ({ ...c, main_color: key })); setResult(null) }} />
                  ))}
                </div>
              </section>
            </>
          )}

          <div className="bench-actions">
            <Magnetic strength={0.18}>
              <button className="run-btn" type="submit" disabled={status !== 'ready' || loading}>
                {loading ? <><LoaderCircle className="spin" size={18} /> Reading</> : <>Predict <ArrowRight size={18} /></>}
              </button>
            </Magnetic>
            <button type="button" className="text-btn" onClick={resetAll}><RotateCcw size={14} /> Reset</button>
          </div>
        </Reveal>

        <Reveal delay={640} className="stage-wrap">
          <aside className={`stage ${loading ? 'is-loading' : ''}`}>
            <div className="stage-bar">
              <span className="eyebrow light">{mode === 'photo' ? 'Photo' : 'Live preview'}</span>
              {needsSpecs && <span className="mono-dim">{form.width_cm || '–'} × {form.depth_cm || '–'} × {form.height_cm || '–'} cm</span>}
            </div>
            <div className="stage-visual">
              {mode === 'photo' && photo
                ? <img src={preview} alt="Selected furniture" className="stage-photo" />
                : mode === 'photo' && !useSpecs
                  ? <div className="stage-empty"><ImagePlus size={34} strokeWidth={1.2} /><span>No photo yet</span></div>
                  : <LiveOutline w={Number(form.width_cm)} d={Number(form.depth_cm)} h={Number(form.height_cm)} busy={loading} />}
              {loading && <span className="scan" aria-hidden="true" />}
            </div>

            <div className="stage-out">
              {loading && <p className="status-line"><LoaderCircle className="spin" size={16} /> Running the saved model on this computer…</p>}
              {!loading && error && (
                <div className="err" role="alert">
                  <AlertCircle size={18} />
                  <div><strong>Prediction could not run</strong><p>{error}</p></div>
                  <button type="button" onClick={clear} aria-label="Dismiss"><X size={15} /></button>
                </div>
              )}
              {!loading && !error && result && <ResultView key={result.prediction + result.confidence} result={result} />}
              {!loading && !error && !result && (
                <p className="idle">Set the details on the left and press <b>Predict</b>. The answer appears here with the model's confidence across all nine types.</p>
              )}
            </div>
          </aside>
        </Reveal>
      </form>
    </div>
  )
}
