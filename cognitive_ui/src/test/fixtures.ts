// Response fixtures shared by component and hook tests, trimmed from docs/API.md.
import type { SampleResponse } from '../api/sample';
import type { SessionInfo } from '../types';

const meta = (date: string) => ({ sensor: 'HLS L30', date, size_bytes: 4180159, line: `HLS L30 · B2–B6 · 30 m · ${date}` });
// Trimmed POST /v1/sample (docs/API.md): both files share one id.
export const SAMPLE: SampleResponse = {
  name: 'kakhovka_hls_l30',
  line: 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09',
  attachments: [
    {
      id: '6774daf7', filename: 'hlsl_20230601.tif', path: '/u/6774daf7_hlsl_20230601.tif',
      size: [768, 768], crs: 'EPSG:32636', bounds: [0, 0, 1, 1],
      preview_url: '/v1/download?path=a.png', metadata: meta('2023-06-01'),
    },
    {
      id: '6774daf7', filename: 'hlsl_20230609.tif', path: '/u/6774daf7_hlsl_20230609.tif',
      size: [768, 768], crs: 'EPSG:32636', bounds: [0, 0, 1, 1],
      preview_url: '/v1/download?path=b.png', metadata: meta('2023-06-09'),
    },
  ],
};

export const GUEST_SESSION: SessionInfo = {
  role: 'guest',
  guest: true,
  analyses: { limit: 5, used: 1, remaining: 4, resets_at: '2026-10-23T06:58:27+00:00' },
  kept_hours: 24,
  expires_at: '2026-10-24T05:58:02+00:00',
  can_download_models: false,
};

export const MEMBER_SESSION: SessionInfo = {
  role: 'member',
  guest: false,
  analyses: { limit: null, used: 0, remaining: null, resets_at: null },
  kept_hours: null,
  expires_at: '2026-10-24T06:00:18+00:00',
  can_download_models: true,
};

export const OPERATOR_SESSION: SessionInfo = { ...MEMBER_SESSION, role: 'operator' };
