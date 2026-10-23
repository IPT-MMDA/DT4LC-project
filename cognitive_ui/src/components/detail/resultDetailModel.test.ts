import { describe, expect, it } from 'vitest';
import type { Capabilities } from '../../api/hooks/useCapabilities';
import type { GeoTIFFFile } from '../../api/hooks/useUpload';
import type { JobRecord } from '../../api/jobs';
import capabilities from '../../test/fixtures/capabilities/offline-member.json';
import lulc from '../../test/fixtures/jobs/lulc.json';
import ndmiChange from '../../test/fixtures/jobs/ndmi-change.json';
import ndviSingle from '../../test/fixtures/jobs/ndvi-single.json';
import ndwiChange from '../../test/fixtures/jobs/ndwi-change.json';
import ndwiChangeV050 from '../../test/fixtures/jobs/ndwi-change-v050.json';
import regionFailed from '../../test/fixtures/jobs/region-failed.json';
import sample from '../../test/fixtures/jobs/sample.json';
import statistics from '../../test/fixtures/jobs/statistics.json';
import type { MessageAttachment } from '../../types';
import { resultCardOf } from '../chat/result/resultModel';
import {
  contextFromJob,
  downloadsOf,
  formatBytes,
  formatDecimal,
  formatSigned,
  inputDates,
  jobInputPaths,
  mapMeta,
  planSummary,
  registryOf,
  regionOfJob,
  resultTitle,
  statisticsTable,
} from './resultDetailModel';

// Fixtures: GET /v1/jobs/{id} from real runs on the bundled sample (guest, no
// assistant); ndwi-change, ndmi-change, statistics and region-failed keep the full
// event log.
const SAMPLE_FILES = sample.attachments.map((a) => ({ ...a, group: { name: sample.name, line: sample.line } })) as MessageAttachment[];
const context = { attachments: SAMPLE_FILES };
const registry = registryOf(capabilities as unknown as Capabilities);
const job = (j: unknown) => j as JobRecord;
const plan = (j: unknown) => planSummary(job(j), registry, SAMPLE_FILES);

describe('plan steps with parameters and times', () => {
  it('NDWI change: both inputs by date, the registry formula with the threshold, the step times', () => {
    const p = plan(ndwiChange);
    expect(p.heading).toBe('4 steps · 2 s');
    expect(p.lines.map((l) => [l.name, l.status, l.params])).toEqual([
      ['Read hlsl_20230601.tif', 'done', 'before · 2023-06-01 · <1 s'],
      ['Read hlsl_20230609.tif', 'done', 'after · 2023-06-09 · <1 s'],
      ['Change Detection · NDWI', 'done', '(Green − NIR)/(Green + NIR) · after − before · threshold 0.2 · 2 s'],
      ['Interpretation', 'done', '<1 s'],
    ]);
  });

  it('NDMI composed: the plan config\'s formula and threshold', () => {
    const p = plan(ndmiChange);
    expect(p.heading).toBe('4 steps · 2 s');
    expect(p.lines[2]).toEqual({ index: 3, name: 'NDMI', status: 'done', params: '(B5 − B6)/(B5 + B6) · after − before · threshold 0.2 · 2 s' });
  });

  it('statistics: no formula, the time alone', () => {
    const p = plan(statistics);
    expect(p.heading).toBe('3 steps · <1 s');
    expect(p.lines.map((l) => l.params)).toEqual(['2023-06-09 · <1 s', '<1 s', '<1 s']);
  });

  it('a single index: the registry formula', () => {
    // ndvi-single keeps only the ready and done events.
    expect(plan(ndviSingle).lines[1].params).toBe('(NIR − Red)/(NIR + Red)');
  });

  it('a proposal: the proposed steps named by the registry, queued', () => {
    const proposal = { ...ndmiChange, status: 'proposal', plan: null, events: ndmiChange.events.slice(0, 3) };
    const p = plan(proposal);
    expect(p.lines.map((l) => [l.name, l.status])).toEqual([
      ['input/file-before', 'queued'],
      ['input/file-after', 'queued'],
      ['algorithms/band-math-change', 'queued'],
      ['post-processing/agent-analysis', 'queued'],
    ]);
    expect(p.lines[2].params).toBe('(B5 − B6)/(B5 + B6) · after − before · threshold 0.2 · queued');
  });

  it('a region job failing at planning has no steps', () => {
    expect(plan(regionFailed)).toEqual({ heading: 'no steps · <1 s', lines: [] });
  });

  it('a Preparing model row has no plan step and is not counted', () => {
    const events = [
      { seq: 1, type: 'planning', data: { phase: 'ready', steps: [
        { index: 1, id: 'input/file', name: 'Read a.tif', kind: 'input' },
        { index: 2, id: 'prepare/prithvi', name: 'Preparing model · Prithvi', kind: 'prepare' },
        { index: 3, id: 'algorithms/ndvi', name: 'NDVI', kind: 'algorithm' },
      ] } },
    ];
    const p = planSummary(job({ status: 'running', events, plan: { steps: [{ uses: 'input/file', binds: {} }, { uses: 'algorithms/ndvi' }] } }), registry, []);
    expect(p.heading).toBe('2 steps');
    expect(p.lines.map((l) => l.params)).toEqual(['queued', 'queued', '(NIR − Red)/(NIR + Red) · queued']);
  });
});

describe('statistics table', () => {
  it('change: per class with area, % of region and mean ΔNDWI, colours from the server', () => {
    const t = statisticsTable(job(ndwiChange), resultCardOf(job(ndwiChange), context));
    expect(t).toEqual({
      kind: 'classes',
      meanLabel: 'Mean ΔNDWI',
      rows: [
        { name: 'Water gained', colour: '#2d5f9e', area: '156.8', percent: '29.5', mean: '+0.81' },
        { name: 'Water lost', colour: '#8c4a2f', area: '0.6', percent: '0.1', mean: '−0.23' },
        { name: 'No change', colour: '#c4cad6', area: '373.5', percent: '70.4', mean: '+0.16' },
      ],
    });
  });

  it('NDMI band math: mean ΔNDMI', () => {
    const t = statisticsTable(job(ndmiChange), resultCardOf(job(ndmiChange), context));
    expect(t?.kind === 'classes' && t.meanLabel).toBe('Mean ΔNDMI');
  });

  it('land cover: classes without a mean column', () => {
    const t = statisticsTable(job(lulc), resultCardOf(job(lulc), context));
    if (t?.kind !== 'classes') throw new Error();
    expect(t.meanLabel).toBeNull();
    expect(t.rows[0]).toEqual({ name: 'Water', colour: '#0077be', area: '204.3', percent: '38.5' });
    expect(t.rows.map((r) => r.name)).toEqual(['Water', 'Snow/Ice', 'Bare Soil', 'Sparse Vegetation', 'Cropland/Grassland', 'Dense Vegetation']);
  });

  it('raster statistics: one row per band code', () => {
    const t = statisticsTable(job(statistics), resultCardOf(job(statistics), context));
    if (t?.kind !== 'bands') throw new Error();
    expect(t.rows.map((r) => r.band)).toEqual(['B2', 'B3', 'B4', 'B5', 'B6']);
    expect(t.rows[3]).toEqual({ band: 'B5', min: '−251', max: '6 957', mean: '1 419', std: '1 221', valid: '589 824' });
  });

  it('a single index: the scalar statistics', () => {
    const t = statisticsTable(job(ndviSingle), resultCardOf(job(ndviSingle), context));
    expect(t).toEqual({
      kind: 'scalars',
      rows: [
        { label: 'Mean', value: '0.17' },
        { label: 'Min', value: '−1' },
        { label: 'Max', value: '1' },
        { label: 'Valid pixels', value: '589 824' },
      ],
    });
  });

  it('a failed job has none', () => {
    expect(statisticsTable(job(regionFailed), null)).toBeNull();
  });

  it('formats areas, shares and means', () => {
    expect(formatDecimal(3570.24)).toBe('3 570.2');
    expect(formatDecimal(0)).toBe('0.0');
    expect(formatDecimal(undefined)).toBe('—');
    expect(formatSigned(0.5249)).toBe('+0.52');
    expect(formatSigned(-0.001)).toBe('0.00');
    expect(formatSigned(null)).toBe('—');
  });
});

describe('downloads', () => {
  it('every file of the NDWI change once, rasters first, with kind and size', () => {
    const d = downloadsOf(job(ndwiChange));
    expect(d.map((f) => `${f.name} · ${f.detail}`)).toEqual([
      'water-change_ndwi_2023-06-01_2023-06-09.tif · GeoTIFF · 2.0 MB',
      'ndwi_2023-06-01.tif · GeoTIFF · 2.0 MB',
      'ndwi_2023-06-09.tif · GeoTIFF · 2.0 MB',
      'water-change_ndwi_2023-06-01_2023-06-09.png · PNG · 1.2 MB',
      'ndwi_2023-06-01.png · PNG · 891 KB',
      'ndwi_2023-06-09.png · PNG · 1.5 MB',
      'water-change_ndwi_2023-06-01_2023-06-09_map.png · PNG · 826 KB',
      'ndwi_2023-06-01_map.png · PNG · 523 KB',
      'ndwi_2023-06-09_map.png · PNG · 1.1 MB',
    ]);
    // The stored path stays; the url names the download.
    expect(d[0].url).toMatch(/^\/v1\/download\?path=.*2_ChangeMap_change\.tif&name=water-change_ndwi_2023-06-01_2023-06-09\.tif$/);
  });

  it('an older job without download_name: the stored file names', () => {
    expect(downloadsOf(job(ndwiChangeV050)).map((f) => f.name)).toEqual([
      '2_ChangeMap_change.tif',
      '2_ChangeMap_index_before.tif',
      '2_ChangeMap_index_after.tif',
      '2_ChangeMap_change_map.png',
      '2_ChangeMap_ndwi_before.png',
      '2_ChangeMap_ndwi_after.png',
    ]);
  });

  it('statistics wrote no file; a ref without size_bytes (older job) has the kind alone', () => {
    expect(downloadsOf(job(statistics))).toEqual([]);
    const old = { result: { execution: { artifacts: { NDVIMap: { rasters: { ndvi: { kind: 'geotiff', path: '/o/1_ndvi.tif', url: '/v1/download?path=x' } } } } } } };
    expect(downloadsOf(job(old))).toEqual([{ name: '1_ndvi.tif', detail: 'GeoTIFF', url: '/v1/download?path=x' }]);
  });

  it('formats sizes 1024-based', () => {
    expect(formatBytes(4180159)).toBe('4.0 MB');
    expect(formatBytes(200)).toBe('1 KB');
    expect(formatBytes(419840)).toBe('410 KB');
    expect(formatBytes(undefined)).toBe('');
  });
});

describe('the map line, title and context', () => {
  it('dates, resolution and CRS', () => {
    expect(mapMeta(job(ndwiChange), context)).toBe('2023-06-01 → 2023-06-09 · 30 m · EPSG:32636');
    expect(mapMeta(job(ndviSingle), context)).toBe('2023-06-09 · 30 m · EPSG:32636');
    // Without the files' metadata: the output's transform and CRS.
    expect(mapMeta(job(ndwiChange), { attachments: [] })).toBe('30 m · EPSG:32636');
    expect(inputDates(job(ndwiChange), context)).toEqual(['2023-06-01', '2023-06-09']);
    expect(inputDates(job(ndwiChange), { attachments: [] })).toEqual(['', '']);
  });

  it('the card title, else the prompt', () => {
    expect(resultTitle(ndwiChange)).toBe('Water change · NDWI difference');
    expect(resultTitle({ id: 'x1', prompt: 'Map NDVI ', card: null })).toBe('Map NDVI');
  });

  it('without a conversation: the library files the job read, the sample as its group', () => {
    expect(jobInputPaths(job(ndwiChange))).toEqual(SAMPLE_FILES.map((f) => f.path));
    const files = sample.attachments.map((a) => ({
      id: 'f7bcb2ae', upload_id: 'f7bcb2ae', filename: a.filename, path: a.path, size: [768, 768], crs: 'EPSG:32636', bounds: [0, 0, 1, 1],
      size_bytes: 4180159, source: 'upload', modified: 1, kind: 'sample', group: { name: 'kakhovka_hls_l30.tif', line: 'HLS L30 · B2–B6' }, metadata: a.metadata,
    })) as GeoTIFFFile[];
    const c = contextFromJob(job(statistics), files);
    expect(c.attachments.map((a) => [a.filename, a.group?.name, a.metadata?.date])).toEqual([['hlsl_20230609.tif', 'kakhovka_hls_l30.tif', '2023-06-09']]);
    expect(c.region).toBeUndefined();
  });

  it('a region job: the echo as the chip region', () => {
    const echo = { bbox: [32.4, 46.47, 32.66, 46.655], name: 'Dnipro floodplain', dataset: 'HLS L30', start_date: '2023-06-01', end_date: '2023-06-09', cloud_max: 20, area_km2: 410.2, line: 'Region · Dnipro floodplain · 410 km²' };
    const r = regionOfJob({ id: 'r1', region: echo });
    expect(r).toMatchObject({ id: 'job-r1', name: 'Dnipro floodplain', dataset: 'HLS L30', area_km2: 410.2, line: echo.line });
    expect(r?.geometry).toEqual({ type: 'Polygon', coordinates: [[[32.4, 46.47], [32.66, 46.47], [32.66, 46.655], [32.4, 46.655], [32.4, 46.47]]] });
    expect(regionOfJob({ id: 'r2', region: null })).toBeUndefined();
  });
});
