import { Niivue, SHOW_RENDER } from '@niivue/niivue'
import { useEffect, useRef, useState } from 'react'

import { useI18n, type Key } from '../lib/i18n'
import { loadImage } from '../lib/nifti'

// Patient view: a see-through brain with the approved lesion in red, rotatable. Values chosen for the
// legacy patient mode (glass opacity tried from 0.12 to 1).
const GLASS_OPACITY = 0.4
const VIEWS = { left: [90, 10], right: [270, 10], front: [180, 10], top: [0, 90] } as const
type View = keyof typeof VIEWS

export default function PatientBrain({ caseId, side }: { caseId: string; side: string }) {
  const { t } = useI18n()
  const canvas = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [status, setStatus] = useState(t('loading'))

  useEffect(() => {
    let cancelled = false
    const nv = new Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: false, isColorbar: false,
                            dragAndDropEnabled: false, multiplanarShowRender: SHOW_RENDER.ALWAYS })
    nvRef.current = nv
    ;(async () => {
      try {
        await nv.attachToCanvas(canvas.current!)
        const [brain, mask] = await Promise.all([
          loadImage(`/me/results/${caseId}/dwi`, 'brain.nii', 'gray', GLASS_OPACITY),
          loadImage(`/me/results/${caseId}/mask`, 'lesion.nii', 'red', 1),
        ])
        if (cancelled) return
        nv.volumes = [brain, mask]
        nv.setVolume(brain, 0)
        nv.setSliceType(nv.sliceTypeRender)
        nv.setCrosshairWidth(0)
        const start: View = side === 'left' ? 'left' : side === 'right' ? 'right' : 'top'
        nv.setRenderAzimuthElevation(...(VIEWS[start] as [number, number]))
        setStatus('')
      } catch (e) {
        if (!cancelled) setStatus(String((e as Error).message ?? e))
      }
    })()
    return () => { cancelled = true; nvRef.current = null }
  }, [caseId, side])

  return (
    <div className="brain">
      <div className="brain-canvas">
        <canvas ref={canvas} aria-label={t('brainAlt')} />
        {status && <div className="vw-status" role="status">{status}</div>}
      </div>
      <div className="views" role="group" aria-label={t('views')}>
        {(Object.keys(VIEWS) as View[]).map((v) => (
          <button key={v} type="button" className="btn-small"
            onClick={() => nvRef.current?.setRenderAzimuthElevation(...(VIEWS[v] as unknown as [number, number]))}>
            {t(`view_${v}` as Key)}
          </button>
        ))}
      </div>
    </div>
  )
}
