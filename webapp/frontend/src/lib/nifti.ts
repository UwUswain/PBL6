import { NVImage } from '@niivue/niivue'

import { fetchAuthed } from './api'

/** Fetch a NIfTI with the bearer token and hand the bytes to NiiVue. */
export async function loadImage(path: string, name: string, colormap: string, opacity = 1): Promise<NVImage> {
  const buffer = await (await fetchAuthed(path)).arrayBuffer()
  // NiiVue detects gzip from the bytes; the name only needs a NIfTI extension.
  return NVImage.loadFromUrl({ url: name, name, buffer, colormap, opacity })
}
