import { describe, expect, it } from 'vitest';
import type { JobRecord } from '../../../api/jobs';
import type { DrawnRegion, Layer } from '../../../store/useAppStore';
import type { MessageAttachment } from '../../../types';
import ndwiChange from '../../../test/fixtures/jobs/ndwi-change.json';
import sample from '../../../test/fixtures/jobs/sample.json';
import { resultCardOf } from '../../chat/result/resultModel';
import {
  activeResultLayer,
  defaultCompareSides,
  layerRows,
  legendOf,
  relativeTime,
  sampleLayerOf,
  samplesFromFiles,
  signedTick,
} from './layers';

const FOOTPRINT: [number, number][] = [[32.388779, 46.662667], [32.689939, 46.663879], [32.691118, 46.456535], [32.391103, 46.455331]];
// The sample pair as the conversation holds it (POST /v1/sample), with BR1's footprint.
const SAMPLE_FILES = sample.attachments.map((a) => ({
  ...a,
  group: { name: sample.name, line: sample.line },
  metadata: { ...a.metadata, footprint_wgs84: FOOTPRINT },
})) as MessageAttachment[];

/** The NDWI change result as the result card's On map adds it (ResultCard Actions). */
function resultLayer(addedAt: string): Layer {
  const model = resultCardOf(ndwiChange as unknown as JobRecord, { attachments: SAMPLE_FILES });
  if (!model || model.kind !== 'change') throw new Error('expected a change card');
  const view = model.views[0];
  return {
    id: 'result-b64b5874-difference', name: view.layerName, type: 'raster', visible: true, opacity: 0.8,
    url: view.raster?.path, bounds: model.bounds, legend: view.legend, source: 'chat', subtitle: 'from chat', addedAt,
  };
}

const region: DrawnRegion = {
  id: 'region-1',
  name: 'Dnipro floodplain · Kherson',
  type: 'bbox',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[33.27, 46.87], [33.55, 46.87], [33.55, 46.69], [33.27, 46.69], [33.27, 46.87]]] },
  },
  bbox: [33.27, 46.69, 33.55, 46.87],
  createdAt: '2026-10-23T10:00:00.000Z',
};

const NOW = Date.parse('2026-10-23T10:00:48Z');

describe('the sample as a true-colour layer', () => {
  it('is the later date of the pair, drawn by /v1/tiles in true colour, with its footprint', () => {
    expect(sampleLayerOf(SAMPLE_FILES)).toEqual({
      id: `sample-${sample.attachments[1].id}-2023-06-09`,
      name: 'kakhovka_hls_l30',
      type: 'raster',
      visible: true,
      opacity: 1,
      url: sample.attachments[1].path,
      source: 'sample',
      subtitle: 'HLS L30 · true colour · 2023-06-09',
      footprint: FOOTPRINT,
      thumbnail: sample.attachments[1].preview_url,
    });
  });

  it('needs the sample: a plain upload or nothing gives no layer', () => {
    expect(sampleLayerOf([])).toBeUndefined();
    expect(sampleLayerOf([{ ...SAMPLE_FILES[0], group: undefined }])).toBeUndefined();
  });
});

describe('the Layers list', () => {
  it('lists the result, the region and the sample, each with its opacity', () => {
    const layers = [sampleLayerOf(SAMPLE_FILES)!, resultLayer('2026-10-23T10:00:00Z')];
    const rows = layerRows(layers, region, 1, NOW);
    expect(rows.map((r) => [r.name, r.subtitle, r.opacity])).toEqual([
      ['Water change · NDWI', 'from chat · 48 s ago', 80],
      ['Dnipro floodplain · Kherson', 'region · 427 km²', 100],
      ['kakhovka_hls_l30', 'HLS L30 · true colour · 2023-06-09', 100],
    ]);
    // The result's swatch is its style's ramp.
    expect(rows[0].swatch).toEqual({ stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'] });
    expect(rows[1].swatch).toEqual({ token: 'region' });
    // The sample's swatch is its preview.
    expect(rows[2].swatch).toEqual({ image: sample.attachments[1].preview_url });
  });

  it("shows each layer's opacity as set, and the region's own", () => {
    const layers = [sampleLayerOf(SAMPLE_FILES)!, { ...resultLayer('2026-10-23T10:00:00Z'), opacity: 0.35 }];
    expect(layerRows(layers, region, 0.6, NOW).map((r) => r.opacity)).toEqual([35, 60, 100]);
    expect(layerRows(layers, null, 1, NOW).map((r) => r.name)).toEqual(['Water change · NDWI', 'kakhovka_hls_l30']);
  });

  it('says how long ago a layer came from chat', () => {
    expect(relativeTime('2026-10-23T10:00:45Z', NOW)).toBe('just now');
    expect(relativeTime('2026-10-23T10:00:00Z', NOW)).toBe('48 s ago');
    expect(relativeTime('2026-10-23T09:55:00Z', NOW)).toBe('5 min ago');
    expect(relativeTime('2026-10-23T07:00:00Z', NOW)).toBe('3 h ago');
  });
});

describe('the legend panel', () => {
  it("comes from the result layer's style and the result's class_statistics", () => {
    expect(legendOf(resultLayer('2026-10-23T10:00:00Z'))).toEqual({
      title: 'Water change · NDWI difference',
      stops: ['#8c4a2f', '#c4cad6', '#2d5f9e'],
      // The style's real range, not the design's ±0.8.
      ticks: ['−0.5', '0', '+0.5'],
      classes: [
        { label: 'lost', tone: 'lost', colour: '#8c4a2f', area: '0.6 km²' },
        { label: 'no change', tone: 'neutral', colour: '#c4cad6', area: '373 km²' },
        { label: 'gained', tone: 'gained', colour: '#2d5f9e', area: '157 km²' },
      ],
    });
  });

  it('an index layer shows its ramp over its range; a layer without a legend shows none', () => {
    const index: Layer = { id: 'x', name: 'NDWI · 2023-06-09', type: 'raster', visible: true, opacity: 0.8, legend: { kind: 'ramp', stops: ['#996633', '#003399'], range: [-0.5, 0.5] } };
    expect(legendOf(index)).toEqual({ title: 'NDWI · 2023-06-09', stops: ['#996633', '#003399'], ticks: ['−0.5', '+0.5'], classes: [] });
    expect(legendOf(sampleLayerOf(SAMPLE_FILES))).toBeUndefined();
    // An older change layer without its style: the tokens and words.
    expect(legendOf({ ...index, legend: { kind: 'change' } })).toMatchObject({ stops: undefined, ticks: ['lost', 'same', 'gained'] });
  });

  it('is about the topmost visible result from chat', () => {
    const older = { ...resultLayer('2026-10-23T09:00:00Z'), id: 'older', name: 'Older' };
    const newer = resultLayer('2026-10-23T10:00:00Z');
    const base = sampleLayerOf(SAMPLE_FILES)!;
    expect(activeResultLayer([base, older, newer])?.id).toBe(newer.id);
    expect(activeResultLayer([base, older, { ...newer, visible: false }])?.id).toBe('older');
    expect(activeResultLayer([base])).toBeUndefined();
  });

  it('ticks carry a real minus and a plus', () => {
    expect([signedTick(-0.8), signedTick(0), signedTick(0.8)]).toEqual(['−0.8', '0', '+0.8']);
  });
});

describe('Compare sides', () => {
  it('puts the result on the left and the true-colour sample on the right', () => {
    const base = sampleLayerOf(SAMPLE_FILES)!;
    const result = resultLayer('2026-10-23T10:00:00Z');
    expect(defaultCompareSides([base, result])).toEqual({ left: result.id, right: base.id });
    expect(defaultCompareSides([result])).toBeNull();
  });
});

describe('the sample from the library', () => {
  it("takes the latest sample pair from GET /v1/files when the conversation hasn't one", () => {
    const file = (upload_id: string, date: string, created_at: string) => ({
      id: `${upload_id}_hlsl_${date.replace(/-/g, '')}`, upload_id, filename: `${upload_id}_hlsl.tif`, path: `/u/${upload_id}_${date}.tif`,
      kind: 'sample' as const, group: { name: 'kakhovka_hls_l30.tif', line: 'HLS L30 · B2–B6' }, metadata: { date, sensor: 'HLS L30' }, created_at,
    });
    const files = [
      file('old', '2023-06-09', '2026-10-22T10:00:00'),
      file('new', '2023-06-01', '2026-10-23T10:00:00'),
      file('new', '2023-06-09', '2026-10-23T10:00:00'),
      { ...file('up', '2023-06-09', '2026-10-23T11:00:00'), kind: 'upload' as const, group: undefined },
    ];
    const pair = samplesFromFiles(files);
    expect(pair.map((a) => a.path)).toEqual(['/u/new_2023-06-01.tif', '/u/new_2023-06-09.tif']);
    expect(sampleLayerOf(pair)).toMatchObject({ id: 'sample-new-2023-06-09', name: 'kakhovka_hls_l30', url: '/u/new_2023-06-09.tif' });
    expect(samplesFromFiles(undefined)).toEqual([]);
  });
});
