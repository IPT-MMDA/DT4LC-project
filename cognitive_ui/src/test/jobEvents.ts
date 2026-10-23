// Job event sequences as the server streams them: docs/API.md "GET /v1/jobs/{job_id}/events",
// and the NDMI proposal, prepare and EE_NOT_CONFIGURED runs recorded against the real API.
import type { JobEvent } from '../types';

let t = 1792735107;
const ev = (seq: number, type: string, data: Record<string, unknown> = {}): JobEvent => ({ seq, type, data: { ...data, t: (t += 0.5) } });

const NDVI_STEPS = [
  { index: 1, id: 'input/file', name: 'Read hlsl_20230601.tif', kind: 'input' },
  { index: 2, id: 'algorithms/ndvi', name: 'NDVI', kind: 'algorithm' },
  { index: 3, id: 'post-processing/agent-analysis', name: 'Interpretation', kind: 'postprocess' },
];

export const NDVI_FOLLOWUPS = ['Use EVI for dense vegetation', 'Map open water with NDWI', 'Classify land cover in this image'];

/** NDVI on the sample, done (docs/API.md). */
export const NDVI_DONE: JobEvent[] = [
  ev(1, 'intent', { text: "I'll compute NDVI for 1 June from the sample." }),
  ev(2, 'planning', { phase: 'started' }),
  ev(3, 'planning', { phase: 'ready', steps: NDVI_STEPS, composed: false }),
  ev(4, 'step_started', { index: 1, id: 'input/file', name: 'Read hlsl_20230601.tif' }),
  ev(5, 'step_finished', { index: 1, id: 'input/file', elapsed_s: 0.0 }),
  ev(6, 'step_started', { index: 2, id: 'algorithms/ndvi', name: 'NDVI' }),
  ev(7, 'step_finished', { index: 2, id: 'algorithms/ndvi', elapsed_s: 0.54 }),
  ev(8, 'step_started', { index: 3, id: 'post-processing/agent-analysis', name: 'Interpretation' }),
  ev(9, 'step_finished', { index: 3, id: 'post-processing/agent-analysis', elapsed_s: 0.0 }),
  ev(10, 'done', { elapsed_s: 0.7, summary: '3 steps · <1 s · NDVI', followups: NDVI_FOLLOWUPS }),
];

export const NDMI_PROPOSAL_TEXT =
  "I don't have a moisture-change algorithm. I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) from B5 and B6 for both dates and difference them — run that?";
export const NDMI_PLAN = {
  flow: 'Map moisture change with NDMI between the two dates',
  steps: [
    { uses: 'input/file-before', binds: { RasterPathBefore: '/u/a.tif' }, config: {} },
    { uses: 'input/file-after', binds: { RasterPathAfter: '/u/b.tif' }, config: {} },
    { uses: 'algorithms/band-math-change', binds: {}, config: { formula: '(B5 - B6) / (B5 + B6)', name: 'NDMI', threshold: 0.2 } },
    { uses: 'post-processing/agent-analysis', binds: {}, config: {} },
  ],
  outputs: ['publish: chat'],
  composed: true,
};

/** Intent → planning started → proposal: the job waits for confirm. */
export const NDMI_PROPOSAL: JobEvent[] = [
  ev(1, 'intent', { text: "I'll compute NDMI for 1 June and 9 June from the sample." }),
  ev(2, 'planning', { phase: 'started' }),
  ev(3, 'proposal', { text: NDMI_PROPOSAL_TEXT, plan: NDMI_PLAN }),
];

/** After POST /v1/jobs/{id}/confirm, on the same stream. */
export const NDMI_CONFIRMED: JobEvent[] = [
  ...NDMI_PROPOSAL,
  ev(4, 'planning', {
    phase: 'ready',
    composed: true,
    steps: [
      { index: 1, id: 'input/file-before', name: 'Read hlsl_20230601.tif', kind: 'input' },
      { index: 2, id: 'input/file-after', name: 'Read hlsl_20230609.tif', kind: 'input' },
      { index: 3, id: 'algorithms/band-math-change', name: 'NDMI', kind: 'algorithm' },
      { index: 4, id: 'post-processing/agent-analysis', name: 'Interpretation', kind: 'postprocess' },
    ],
  }),
  ev(5, 'step_started', { index: 1, id: 'input/file-before', name: 'Read hlsl_20230601.tif' }),
  ev(6, 'step_finished', { index: 1, id: 'input/file-before', elapsed_s: 0.0 }),
  ev(7, 'step_started', { index: 2, id: 'input/file-after', name: 'Read hlsl_20230609.tif' }),
  ev(8, 'step_finished', { index: 2, id: 'input/file-after', elapsed_s: 0.0 }),
  ev(9, 'step_started', { index: 3, id: 'algorithms/band-math-change', name: 'NDMI' }),
  ev(10, 'step_finished', { index: 3, id: 'algorithms/band-math-change', elapsed_s: 1.19 }),
  ev(11, 'step_started', { index: 4, id: 'post-processing/agent-analysis', name: 'Interpretation' }),
  ev(12, 'step_finished', { index: 4, id: 'post-processing/agent-analysis', elapsed_s: 0.0 }),
  ev(13, 'done', {
    elapsed_s: 2.5,
    summary: '4 steps · 2 s · NDMI',
    followups: ['Redo the moisture map with a 0.1 threshold', 'Compare with NDWI water change between the same dates', 'What else can you do with this data?'],
  }),
];

export const EE_FIXES = [
  { label: 'Use my own image', kind: 'upload' },
  { label: 'Try the Kakhovka sample', kind: 'sample' },
];

/** A region job without Earth Engine: refused at planning, no skeleton. */
export const EE_FAILED: JobEvent[] = [
  ev(1, 'intent', { text: "I'll compute NDWI for 1 June and 9 June from Dnipro floodplain, then difference them to map where water changed." }),
  ev(2, 'planning', { phase: 'started' }),
  ev(3, 'failed', {
    code: 'EE_NOT_CONFIGURED',
    message: "Earth Engine isn't configured on this server, so I can't fetch imagery for this region.",
    fixes: EE_FIXES,
    raw: "Earth Engine isn't configured on this server, so I can't fetch imagery for this region.",
  }),
];

/** NDVI that fails in step 2 with every fix kind. */
export const NDVI_FAILED_AT_2: JobEvent[] = [
  ...NDVI_DONE.slice(0, 6),
  ev(7, 'failed', {
    code: 'INTERNAL',
    message: 'Something went wrong while computing NDVI.',
    step_index: 2,
    raw: 'Traceback: ValueError: boom',
    fixes: [
      { label: 'Retry', kind: 'retry', params: { mode: 'template' } },
      { label: 'Map open water with NDWI', kind: 'prompt', prompt: 'Map open water with NDWI' },
      { label: 'Pick a region on the map', kind: 'open_map' },
    ],
  }),
];

/** Cancel mid-run (docs/API.md): the running step is not finished, nothing follows. */
export const NDVI_CANCELLED: JobEvent[] = [...NDVI_DONE.slice(0, 6), ev(7, 'cancelled', { at_step: 2 })];

/** A member's Prithvi run with the weights missing: the Preparing model row mid-download. */
export const PRITHVI_PREPARING: JobEvent[] = [
  ev(1, 'intent', { text: "I'll run Prithvi MAE Reconstruction on hls6_20230609.tif." }),
  ev(2, 'planning', { phase: 'started' }),
  ev(3, 'planning', {
    phase: 'ready',
    composed: false,
    steps: [
      { index: 1, id: 'input/file', name: 'Read hls6_20230609.tif', kind: 'input' },
      { index: 2, id: 'prepare/prithvi-eo-v1-100m', name: 'Preparing model · Prithvi EO v1 (100M)', kind: 'prepare', model_id: 'prithvi-eo-v1-100m' },
      { index: 3, id: 'models/prithvi-reconstruction', name: 'Prithvi MAE Reconstruction', kind: 'model' },
      { index: 4, id: 'post-processing/agent-analysis', name: 'Interpretation', kind: 'postprocess' },
    ],
  }),
  ev(4, 'step_started', { index: 1, id: 'input/file', name: 'Read hls6_20230609.tif' }),
  ev(5, 'step_finished', { index: 1, id: 'input/file', elapsed_s: 0.0 }),
  ev(6, 'step_started', { index: 2, id: 'prepare/prithvi-eo-v1-100m', name: 'Preparing model · Prithvi EO v1 (100M)' }),
  ev(7, 'step_progress', { index: 2, fraction: 0.0, message: 'Downloading Prithvi EO v1 (100M) · 450 MB', bytes: 0, total_bytes: 471859200, eta_s: 75 }),
  ev(8, 'step_progress', { index: 2, fraction: 0.111, message: 'Downloading Prithvi EO v1 (100M) · 450 MB', bytes: 52428801, total_bytes: 471859200, eta_s: 56 }),
];
