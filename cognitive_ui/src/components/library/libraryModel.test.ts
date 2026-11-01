import { describe, expect, it } from 'vitest';
import type { GeoTIFFFile } from '../../api/hooks/useUpload';
import type { JobCard, JobSummary } from '../../api/library';
import type { Conversation } from '../../types';
import {
  analysisContext,
  DEFAULT_FILTERS,
  filterRows,
  formatRunTime,
  libraryRows,
  rowAction,
  rowWhen,
  typeOptions,
  whenLabel,
  type AnalysisRow,
  type FileRow,
} from './libraryModel';

// Local times on 23 Oct 2026, as the browser shows them.
const NOW = new Date(2026, 9, 23, 15, 0).getTime();
// The server's naive local ISO times, e.g. "2026-10-23T14:02:00".
const at = (day: number, h: number, m = 0) => `2026-10-${String(day).padStart(2, '0')}T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00`;

const card = (over: Partial<JobCard>): JobCard => ({
  title: 'Water change · NDWI difference',
  analysis: { id: 'algorithms/change-detection', label: 'Change detection' },
  kind: 'change',
  thumbnail_url: '/v1/download?path=%2Fo%2Fndwi_bare.png',
  duration_s: 48.2,
  error: null,
  ...over,
});
const job = (id: string, over: Partial<JobSummary>): JobSummary => ({ id, status: 'completed', prompt: '', created_at: at(23, 14, 2), ...over });

const NDWI = job('a7f3e2', { prompt: 'Compare water before and after the dam breach', card: card({}) });
const STATS = job('b1', {
  prompt: 'Give me statistics for every band',
  created_at: at(22, 9),
  card: card({ title: 'Raster statistics · 2023-06-09', analysis: { id: 'algorithms/statistics', label: 'Statistics' }, kind: 'statistics', thumbnail_url: null, duration_s: 6 }),
});
const FAILED = job('c2', {
  status: 'failed',
  prompt: 'Compare water before and after the dam breach',
  created_at: at(23, 14, 6),
  card: card({ title: 'Compare water before and after the dam breach', thumbnail_url: null, duration_s: 0.4, error: { code: 'EE_NOT_CONFIGURED', message: "Earth Engine isn't configured on this server" } }),
});
const CANCELLED = job('d3', { status: 'cancelled', prompt: 'calculate ndvi', created_at: at(23, 13), card: card({ title: 'NDVI · 2023-06-01', analysis: { id: 'algorithms/ndvi', label: 'NDVI' }, kind: 'index', duration_s: 75 }) });
const OLD = job('e4', { prompt: 'calculate ndvi', created_at: at(1, 10), card: null });
const RUNNING = job('f5', { status: 'running', prompt: 'Map land cover', created_at: at(23, 14, 30), card: null });

const group = { name: 'kakhovka_hls_l30.tif', line: 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636' };
const file = (over: Partial<GeoTIFFFile>): GeoTIFFFile => ({
  id: 'x', filename: 'x.tif', path: '/u/x.tif', size: [1, 1], crs: null, bounds: [0, 0, 1, 1], size_bytes: 1, source: 'upload', modified: 0, ...over,
});
const SAMPLE_FILES = ['20230601', '20230609'].map((d) =>
  file({ id: `6774daf7_hlsl_${d}`, upload_id: '6774daf7', filename: `6774daf7_hlsl_${d}.tif`, path: `/u/6774daf7_hlsl_${d}.tif`, kind: 'sample', group, preview_url: `/v1/download?path=${d}.png`, created_at: at(23, 14) }),
);
const UPLOAD = file({
  id: '35968520_forest', upload_id: '35968520', filename: '35968520_forest.tif', path: '/u/35968520_forest.tif', kind: 'upload',
  metadata: { line: 'Sentinel-2 · B2 B3 B4 B8 · 2024-07-11', size_label: '1.2 MB' }, created_at: at(20, 8),
});
const EXPORT = file({ id: 'gee_ndvi', filename: 'gee_ndvi.tif', path: '/x/gee_ndvi.tif', source: 'export', kind: 'export', created_at: at(23, 14, 3) });

const conversation = (id: string, title: string, messages: Conversation['messages']): Conversation => ({ id, title, createdAt: at(23, 14), updatedAt: at(23, 14), messages });
const sampleAttachment = (d: string) => ({ id: '6774daf7', filename: `hlsl_${d}.tif`, path: `/u/6774daf7_hlsl_${d}.tif`, group: { name: 'kakhovka_hls_l30', line: '' } });
const KAKHOVKA = conversation('k1', 'Kakhovka reservoir — water change', [
  { id: 'u1', kind: 'user', text: 'Compare water before and after the dam breach', createdAt: at(23, 14), attachments: [sampleAttachment('20230601'), sampleAttachment('20230609')] },
  { id: 'a1', kind: 'analysis', jobId: 'a7f3e2', status: 'done', text: '', events: [], createdAt: at(23, 14) },
  { id: 'u2', kind: 'user', text: 'Give me statistics for every band', createdAt: at(23, 14) },
  { id: 'a2', kind: 'analysis', jobId: 'b1', status: 'done', text: '', events: [], createdAt: at(23, 14) },
]);
const SECOND = conversation('k2', 'Sample again', [
  { id: 'u3', kind: 'user', text: 'calculate ndvi', createdAt: at(23, 13), attachments: [sampleAttachment('20230609')] },
  { id: 'a3', kind: 'analysis', jobId: 'd3', status: 'cancelled', text: '', events: [], createdAt: at(23, 13) },
]);
const CONVERSATIONS = [KAKHOVKA, SECOND];

const rows = () => libraryRows([NDWI, STATS, FAILED, CANCELLED, OLD, RUNNING], [...SAMPLE_FILES, UPLOAD, EXPORT], CONVERSATIONS);
const find = (key: string) => rows().find((r) => r.key === key)!;

describe('Library rows: analyses and files merged', () => {
  it('sorts analyses and files together, newest first, the sample pair as one row', () => {
    expect(rows().map((r) => r.key)).toEqual([
      'job-f5', 'job-c2', 'file-/x/gee_ndvi.tif', 'job-a7f3e2', 'sample-6774daf7', 'job-d3', 'job-b1', 'file-/u/35968520_forest.tif', 'job-e4',
    ]);
  });

  it('names an analysis by its card: title, quoted prompt, label, run time, conversation', () => {
    expect(find('job-a7f3e2')).toMatchObject({
      kind: 'analysis', title: 'Water change · NDWI difference', subtitle: '"Compare water before and after the dam breach"',
      type: 'Change detection', status: 'done', duration: '48 s', conversation: 'Kakhovka reservoir — water change', conversationId: 'k1',
      thumbnail: '/v1/download?path=%2Fo%2Fndwi_bare.png',
    });
  });

  it("shows a failed job's error message under its title", () => {
    expect(find('job-c2')).toMatchObject({ status: 'failed', subtitle: "Earth Engine isn't configured on this server", duration: '<1 s', conversation: '—' });
  });

  it('falls back to the prompt without a card (old rows, running jobs)', () => {
    expect(find('job-e4')).toMatchObject({ title: 'calculate ndvi', subtitle: '', type: '', duration: '', status: 'done', action: null, thumbnail: null });
    expect(find('job-f5')).toMatchObject({ title: 'Map land cover', status: 'running', action: null });
  });

  it('names a proposal job a proposal, not queued', () => {
    const [row] = libraryRows([job('p1', { status: 'proposal', prompt: 'Map the flood', card: null })], [], []);
    expect(row).toMatchObject({ kind: 'analysis', status: 'proposal', title: 'Map the flood' });
  });

  it('names files by group or file name and line, and counts the conversations that attach them', () => {
    expect(find('sample-6774daf7')).toMatchObject({
      kind: 'file', fileKind: 'sample', title: 'kakhovka_hls_l30.tif', subtitle: group.line, type: 'File · sample',
      conversation: '2 conversations', deleteId: '6774daf7', thumbnail: '/v1/download?path=20230601.png',
    });
    expect((find('sample-6774daf7') as FileRow).files).toHaveLength(2);
    expect(find('file-/u/35968520_forest.tif')).toMatchObject({
      title: 'forest.tif', subtitle: 'Sentinel-2 · B2 B3 B4 B8 · 2024-07-11 · 1.2 MB', type: 'File · upload', conversation: '—', deleteId: '35968520',
    });
    expect(find('file-/x/gee_ndvi.tif')).toMatchObject({ type: 'File · export', deleteId: null });
  });

  it('a file attached in one conversation shows its title', () => {
    const [row] = libraryRows([], SAMPLE_FILES, [KAKHOVKA]);
    expect(row.conversation).toBe('Kakhovka reservoir — water change');
  });
});

describe('row action per kind', () => {
  it('Map for done rasters, CSV for statistics, Retry for failures, Chat for files, ↓ for exports', () => {
    expect(Object.fromEntries(rows().map((r) => [r.key, r.action]))).toEqual({
      'job-a7f3e2': 'map',
      'job-b1': 'csv',
      'job-c2': 'retry',
      'job-d3': null,
      'job-e4': null,
      'job-f5': null,
      'sample-6774daf7': 'chat',
      'file-/u/35968520_forest.tif': 'chat',
      'file-/x/gee_ndvi.tif': 'download',
    });
  });

  it.each(['change', 'index', 'classification', 'fields', 'composed', 'image'] as const)('a done %s result goes on the map', (kind) => {
    expect(rowAction({ kind: 'analysis', status: 'done', job: job('z', { card: card({ kind }) }) })).toBe('map');
  });
});

describe('times', () => {
  it('formats run times like the design', () => {
    expect([0.4, 6, 48.2, 115, 120, null].map(formatRunTime)).toEqual(['<1 s', '6 s', '48 s', '1 m 55 s', '2 m', '']);
  });

  it('says today with the time, yesterday, else the date; the sample is bundled', () => {
    expect(whenLabel(new Date(2026, 9, 23, 14, 2).getTime(), NOW)).toBe('today 14:02');
    expect(whenLabel(new Date(2026, 9, 22, 23, 59).getTime(), NOW)).toBe('yesterday');
    expect(whenLabel(new Date(2026, 9, 1, 10).getTime(), NOW)).toBe('2026-10-01');
    expect(rowWhen(find('sample-6774daf7'), NOW)).toBe('bundled');
    expect(rowWhen(find('job-b1'), NOW)).toBe('yesterday');
  });
});

describe('filters', () => {
  const keys = (filters: Partial<typeof DEFAULT_FILTERS>) => filterRows(rows(), { ...DEFAULT_FILTERS, ...filters }, NOW).map((r) => r.key);

  it('shows the last 7 days by default', () => {
    expect(keys({})).not.toContain('job-e4');
    expect(keys({})).toHaveLength(8);
    expect(keys({ since: 'all' })).toHaveLength(9);
  });

  it('All / Analyses / Files', () => {
    expect(keys({ show: 'analyses' })).toEqual(['job-f5', 'job-c2', 'job-a7f3e2', 'job-d3', 'job-b1']);
    expect(keys({ show: 'files' })).toEqual(['file-/x/gee_ndvi.tif', 'sample-6774daf7', 'file-/u/35968520_forest.tif']);
  });

  it('status chips keep analyses with any of the chosen statuses', () => {
    expect(keys({ statuses: ['done'] })).toEqual(['job-a7f3e2', 'job-b1']);
    expect(keys({ statuses: ['failed', 'cancelled'] })).toEqual(['job-c2', 'job-d3']);
  });

  it('Type keeps one analysis label or file type', () => {
    expect(keys({ type: 'Statistics' })).toEqual(['job-b1']);
    expect(keys({ type: 'File · sample' })).toEqual(['sample-6774daf7']);
  });

  it('Today, 30 days', () => {
    expect(keys({ since: 'today' })).toEqual(['job-f5', 'job-c2', 'file-/x/gee_ndvi.tif', 'job-a7f3e2', 'sample-6774daf7', 'job-d3']);
    expect(keys({ since: '30d' })).toHaveLength(9);
  });

  it('offers the analysis labels in the list, then its file types', () => {
    expect(typeOptions(rows())).toEqual(['Change detection', 'NDVI', 'Statistics', 'File · sample', 'File · upload', 'File · export']);
  });
});

describe('analysisContext', () => {
  it("finds the job's conversation, the prompt that asked for it and the files it was about", () => {
    const { conversation, prompt, context } = analysisContext(CONVERSATIONS, 'b1');
    expect(conversation?.id).toBe('k1');
    expect(prompt).toBe('Give me statistics for every band');
    expect(context.attachments.map((a) => a.path)).toEqual(['/u/6774daf7_hlsl_20230601.tif', '/u/6774daf7_hlsl_20230609.tif']);
  });

  it('has nothing for a job from another browser', () => {
    expect(analysisContext(CONVERSATIONS, 'nope')).toEqual({ conversation: null, prompt: '', context: { attachments: [] } });
    expect((find('job-c2') as AnalysisRow).conversationId).toBeNull();
  });
});
