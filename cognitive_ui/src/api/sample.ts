import apiClient from './client';
import type { Attachment } from '../types';
import type { UploadResponse } from './hooks/useUpload';

/** POST /v1/sample: the bundled Kakhovka pair as the caller's uploads (docs/API.md). */
export interface SampleResponse {
  name: string;  // "kakhovka_hls_l30"
  line: string;  // "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09"
  attachments: UploadResponse[];  // oldest first; both share one `id`, so key them by `path`
}

/** Idempotent per session; a server without the sample answers 404 `SAMPLE_NOT_BUNDLED`. */
export function postSample(): Promise<SampleResponse> {
  return apiClient.post<undefined, SampleResponse>('/v1/sample');
}

/** The sample as composer attachments: one group (one chip), metadata kept for jobs and chat. */
export function sampleAttachments(sample: SampleResponse): Attachment[] {
  const group = { name: sample.name, line: sample.line };
  return sample.attachments.map((a) => ({
    id: a.id,
    filename: a.filename,
    path: a.path,
    mime_type: 'image/tiff',
    size_bytes: a.metadata?.size_bytes,
    preview_url: a.preview_url,
    metadata: a.metadata,
    group,
  }));
}
