import { Niivue, NVImage, SHOW_RENDER } from '@niivue/niivue'
import { useCallback, useEffect, useRef, useState } from 'react'

import { api, ApiError } from '../lib/api'
import { loadImage } from '../lib/nifti'
import { useI18n, type Key } from '../lib/i18n'

export type Modality = 'dwi' | 'adc' | 'flair'
type Panel = 'coronal' | 'sagittal' | 'axial' | 'render'
const PANELS: Panel[] = ['coronal', 'sagittal', 'axial', 'render']
// Must match .vw-cells in index.css: gap between cells, header strip, inner padding (px).
const GAP = 10, HEAD = 31, PAD = 6

interface Props {
  caseId: string
  modalities: Modality[]
  maskId?: string | null
  /** Doctor may edit (an AI mask exists, not approved, no job running). */
  editable?: boolean
  /** Called after an edit was saved as a new version. */
  onEditSaved?: () => void
}

/** Axial slices (in the mask's own voxel grid) that contain lesion, with the in-slice centre of each. */
function lesionSlices(mask: NVImage): [number, number, number][] {
  const dims = mask.hdr!.dims
  const [nx, ny, nz] = [dims[1], dims[2], dims[3]]
  const img = mask.img as ArrayLike<number>
  const out: [number, number, number][] = []
  for (let z = 0; z < nz; z++) {
    let n = 0, sx = 0, sy = 0
    const base = z * nx * ny
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (img[base + y * nx + x] > 0) { n++; sx += x; sy += y }
      }
    }
    if (n) out.push([sx / n, sy / n, z])
  }
  return out  // shortcut: assumes the third voxel axis is roughly axial (true for ISLES); reorient for oblique uploads
}

export default function Viewer({ caseId, modalities, maskId, editable = false, onEditSaved }: Props) {
  const { t } = useI18n()
  const canvas = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const cache = useRef<Partial<Record<Modality, NVImage>>>({})
  const maskRef = useRef<NVImage | null>(null)
  const slicesRef = useRef<[number, number, number][]>([])
  const maxRef = useRef<Panel | null>(null)
  const [bg, setBg] = useState<Modality>(modalities.includes('dwi') ? 'dwi' : modalities[0])
  const [showMask, setShowMask] = useState(true)
  const [opacity, setOpacity] = useState(0.5)
  const [maxPanel, setMaxPanel] = useState<Panel | null>(null)
  const [radiological, setRadiological] = useState(false)
  const [loc, setLoc] = useState('')
  const [status, setStatus] = useState<string>('')
  const [lesionInfo, setLesionInfo] = useState({ index: -1, count: 0 })
  const [editing, setEditing] = useState(false)
  const [tool, setTool] = useState<'draw' | 'erase'>('draw')
  const [pen, setPen] = useState(2)
  const [liveMl, setLiveMl] = useState<number | null>(null)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [editError, setEditError] = useState('')

  const updateLive = useCallback(() => {
    const nv = nvRef.current, mask = maskRef.current
    if (!nv?.drawBitmap || !mask) { setLiveMl(null); return }
    let n = 0
    for (const v of nv.drawBitmap) if (v) n++
    const px = mask.hdr!.pixDims
    setLiveMl((n * Math.abs(px[1] * px[2] * px[3])) / 1000)
  }, [])

  const layout = useCallback((nv: Niivue, max: Panel | null) => {
    const W = nv.canvas!.clientWidth || 1, H = nv.canvas!.clientHeight || 1
    const type = { coronal: nv.sliceTypeCoronal, sagittal: nv.sliceTypeSagittal, axial: nv.sliceTypeAxial, render: nv.sliceTypeRender }
    const cells: [number, number, Panel][] = max ? [[0, 0, max]] : [[0, 0, 'coronal'], [1, 0, 'sagittal'], [0, 1, 'axial'], [1, 1, 'render']]
    const cw = max ? W : (W - GAP) / 2, ch = max ? H : (H - GAP) / 2
    nv.setCustomLayout(cells.map(([col, row, p]) => ({
      sliceType: type[p],
      position: [(col * (cw + GAP) + PAD) / W, (row * (ch + GAP) + HEAD) / H, (cw - 2 * PAD) / W, (ch - HEAD - PAD) / H],
    })))
  }, [])

  // Create NiiVue once.
  useEffect(() => {
    const nv = new Niivue({
      backColor: [0, 0, 0, 0],  // transparent: the black cells underneath draw the panels
      show3Dcrosshair: true,
      crosshairColor: [1, 0.78, 0.25, 0.75],
      crosshairWidth: 0.75,
      isColorbar: false,
      dragAndDropEnabled: false,
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
    })
    nvRef.current = nv
    if (import.meta.env.DEV) (window as unknown as { __nv?: Niivue }).__nv = nv  // dev-only hook for browser tests
    let observer: ResizeObserver | null = null
    nv.attachToCanvas(canvas.current!).then(() => {
      nv.canvas!.parentElement!.style.backgroundColor = ''  // NiiVue paints the parent black
      nv.setSliceType(nv.sliceTypeMultiplanar)
      nv.onLocationChange = (d) => setLoc((d as { string?: string }).string ?? '')
      nv.onDrawingChanged = () => updateLive()
      observer = new ResizeObserver(() => { if (nv.volumes.length) layout(nv, maxRef.current) })
      observer.observe(nv.canvas!)
    })
    return () => { observer?.disconnect(); nvRef.current = null; cache.current = {}; maskRef.current = null }
  }, [layout, updateLive])

  useEffect(() => {
    maxRef.current = maxPanel
    const nv = nvRef.current
    if (nv?.volumes.length) layout(nv, maxPanel)
  }, [maxPanel, layout])

  // Load the background and the mask whenever the selection changes.
  useEffect(() => {
    let cancelled = false
    const nv = nvRef.current
    if (!nv) return
    ;(async () => {
      setStatus(t('loading'))
      try {
        cache.current[bg] ??= await loadImage(`/cases/${caseId}/volumes/${bg}/file`, `${bg}.nii`, 'gray')
        if (maskId && maskRef.current?.name !== `mask-${maskId}.nii`) {
          maskRef.current = await loadImage(`/masks/${maskId}/file`, `mask-${maskId}.nii`, 'red', opacity)
          slicesRef.current = lesionSlices(maskRef.current)
          setLesionInfo({ index: -1, count: slicesRef.current.length })
        }
        if (!maskId) { maskRef.current = null; slicesRef.current = []; setLesionInfo({ index: -1, count: 0 }) }
        if (cancelled) return
        nv.volumes = [cache.current[bg]!, ...(maskRef.current ? [maskRef.current] : [])]
        nv.setVolume(cache.current[bg]!, 0)
        layout(nv, maxRef.current)
        setStatus('')
      } catch (e) {
        if (!cancelled) setStatus(String((e as Error).message ?? e))
      }
    })()
    return () => { cancelled = true }
  }, [caseId, bg, maskId, layout])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const nv = nvRef.current
    if (nv && maskRef.current && nv.volumes.includes(maskRef.current)) {
      // While editing, the drawing layer shows the mask; the overlay copy would double it.
      nv.setOpacity(nv.volumes.indexOf(maskRef.current), showMask && !editing ? opacity : 0)
    }
  }, [showMask, opacity, lesionInfo.count, bg, editing])

  // Editing starts once the DWI background is loaded: the drawing must share the mask's (DWI) grid.
  useEffect(() => {
    const nv = nvRef.current
    if (!editing || !nv || bg !== 'dwi' || status || !maskRef.current || nv.drawBitmap) return
    if (!nv.loadDrawing(maskRef.current)) { setEditing(false); setEditError(t('errEdit')); return }
    nv.setDrawingEnabled(true)
    updateLive()
  }, [editing, bg, status, updateLive, t])

  useEffect(() => {
    const nv = nvRef.current
    if (!nv || !editing) return
    nv.setPenValue(tool === 'erase' ? 0 : 1, false)
    nv.opts.penSize = pen
  }, [tool, pen, editing, liveMl === null])  // eslint-disable-line react-hooks/exhaustive-deps

  const stopEditing = useCallback(() => {
    const nv = nvRef.current
    if (nv) { nv.setDrawingEnabled(false); nv.closeDrawing() }
    setEditing(false)
    setLiveMl(null)
    setNote('')
  }, [])

  useEffect(() => { if (!editable && editing) stopEditing() }, [editable, editing, stopEditing])

  function startEditing() {
    setEditError('')
    setBg('dwi')
    setEditing(true)
  }

  async function saveEdit() {
    const nv = nvRef.current
    if (!nv) return
    setSaving(true)
    setEditError('')
    try {
      const bytes = await nv.saveImage({ filename: '', isSaveDrawing: true, volumeByIndex: 0 })
      if (!(bytes instanceof Uint8Array)) throw new Error(t('errEdit'))
      const body = new FormData()
      body.append('file', new Blob([bytes as BlobPart]), 'edited.nii')
      if (note.trim()) body.append('note', note.trim())
      await api(`/cases/${caseId}/masks`, { method: 'POST', body })
      stopEditing()
      onEditSaved?.()
    } catch (e) {
      setEditError(e instanceof ApiError ? e.message : String((e as Error).message ?? e))
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => { nvRef.current?.setRadiologicalConvention(radiological) }, [radiological])

  const jumpLesion = useCallback((step: 1 | -1) => {
    const nv = nvRef.current, mask = maskRef.current, slices = slicesRef.current
    if (!nv || !mask || !slices.length) return
    setLesionInfo((s) => {
      const index = (s.index + step + slices.length) % slices.length
      // Raw voxel -> world mm with the file's affine; NiiVue reorients images, so raw voxels cannot go to frac directly.
      const [i, j, k] = slices[index]
      const A = mask.hdr!.affine
      const mm = [0, 1, 2].map((r) => A[r][0] * i + A[r][1] * j + A[r][2] * k + A[r][3])
      nv.scene.crosshairPos = nv.mm2frac(mm as unknown as Parameters<Niivue['mm2frac']>[0], 0, true)
      nv.drawScene()
      return { ...s, index }
    })
  }, [])

  function onKey(e: React.KeyboardEvent) {
    const nv = nvRef.current
    if (!nv) return
    if (editing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); nv.drawUndo(); updateLive(); return }
    if (e.altKey || e.ctrlKey || e.metaKey) return
    const keys: Record<string, () => void> = {
      ...(editing ? {
        e: () => setTool((x) => (x === 'draw' ? 'erase' : 'draw')),
        '[': () => setPen((x) => Math.max(1, x - 1)),
        ']': () => setPen((x) => Math.min(10, x + 1)),
      } : {}),
      ArrowUp: () => nv.moveCrosshairInVox(0, 0, 1),
      ArrowDown: () => nv.moveCrosshairInVox(0, 0, -1),
      n: () => jumpLesion(1),
      p: () => jumpLesion(-1),
      m: () => setShowMask((v) => !v),
      Escape: () => setMaxPanel(null),
      '0': () => setMaxPanel(null),
      '1': () => setMaxPanel('coronal'),
      '2': () => setMaxPanel('sagittal'),
      '3': () => setMaxPanel('axial'),
      '4': () => setMaxPanel('render'),
    }
    const fn = keys[e.key] ?? keys[e.key.toLowerCase()]
    if (fn) { e.preventDefault(); fn() }
  }

  return (
    <div className="viewer-wrap">
      <div className="viewer-tools" role="toolbar" aria-label={t('viewerTools')}>
        <div className="seg" role="radiogroup" aria-label={t('background')}>
          {(['dwi', 'adc', 'flair'] as Modality[]).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={bg === m} disabled={!modalities.includes(m) || editing} onClick={() => setBg(m)}>
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        {maskId && (
          <>
            <label className="check"><input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> {t('showMask')}</label>
            <label className="inline">{t('opacity')}
              <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
            <div className="lesion-nav">
              <button type="button" className="btn-small" onClick={() => jumpLesion(-1)} disabled={!lesionInfo.count} title="P">‹</button>
              <span className="small">{t('lesionSlices', { i: lesionInfo.index + 1 || '–', n: lesionInfo.count })}</span>
              <button type="button" className="btn-small" onClick={() => jumpLesion(1)} disabled={!lesionInfo.count} title="N">›</button>
            </div>
          </>
        )}
        <label className="check"><input type="checkbox" checked={radiological} onChange={(e) => setRadiological(e.target.checked)} /> {t('radiological')}</label>
        {maskId && !editing && (
          <button type="button" className="btn-small" onClick={startEditing} disabled={!editable} title={editable ? '' : t('editLocked')}>{t('editStart')}</button>
        )}
      </div>

      {editing && (
        <div className="edit-bar" role="toolbar" aria-label={t('editStart')}>
          <div className="seg" role="radiogroup" aria-label={t('editTool')}>
            <button type="button" role="radio" aria-checked={tool === 'draw'} onClick={() => setTool('draw')}>{t('toolDraw')}</button>
            <button type="button" role="radio" aria-checked={tool === 'erase'} onClick={() => setTool('erase')}>{t('toolErase')}</button>
          </div>
          <label className="inline">{t('penSize')} {pen}
            <input type="range" min={1} max={10} value={pen} onChange={(e) => setPen(Number(e.target.value))} />
          </label>
          <button type="button" className="btn-small" onClick={() => { nvRef.current?.drawUndo(); updateLive() }}>{t('undo')}</button>
          <strong className="live">{liveMl === null ? t('loading') : t('editLive', { v: liveMl.toFixed(2) })}</strong>
          <input className="edit-note" placeholder={t('editNote')} aria-label={t('editNote')} value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} />
          <button type="button" className="btn-primary" onClick={saveEdit} disabled={saving || liveMl === null}>{saving ? t('saving') : t('saveEdit')}</button>
          <button type="button" className="btn-small" onClick={stopEditing} disabled={saving}>{t('cancel')}</button>
        </div>
      )}
      {editError && <p className="form-error" role="alert">{editError}</p>}

      <div className="vw" tabIndex={0} onKeyDown={onKey} aria-label={t('viewerHelp')}>
        <div className={`vw-cells ${maxPanel ? 'single' : ''}`} aria-hidden="true">
          {(maxPanel ? [maxPanel] : PANELS).map((p) => (
            <div key={p} className={`vw-cell ${p}`}><span>{t(`panel_${p}` as Key)}</span></div>
          ))}
        </div>
        <canvas ref={canvas} />
        <div className={`vw-cells vw-max ${maxPanel ? 'single' : ''}`} role="group" aria-label={t('maximize')}>
          {(maxPanel ? [maxPanel] : PANELS).map((p) => (
            <div key={p} className="vw-cell-hit">
              <button type="button" className="btn-max" onClick={() => setMaxPanel(maxPanel ? null : p)}
                aria-label={maxPanel ? t('restore') : `${t('maximize')}: ${t(`panel_${p}` as Key)}`}>
                {maxPanel ? '⤡' : '⤢'}
              </button>
            </div>
          ))}
        </div>
        {status && <div className="vw-status" role="status">{status}</div>}
        <div className="vw-loc">{loc}</div>
      </div>
      <p className="muted small">{t(editing ? 'editHelp' : 'viewerHelp')}</p>
    </div>
  )
}
