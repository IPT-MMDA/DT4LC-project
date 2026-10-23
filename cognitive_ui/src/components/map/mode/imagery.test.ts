import { describe, expect, it } from 'vitest';
import type { Capabilities } from '../../../api/hooks/useCapabilities';
import { toChatRegion } from '../../../api/chat';
import type { DrawnRegion } from '../../../store/useAppStore';
import { regionLine } from '../../chat/regionChip';
import {
  bulkFetchRequest,
  defaultImagery,
  earthEngineStatus,
  fetchPeriods,
  formatCentroid,
  regionFromImagery,
  regionProblem,
  toggleBand,
  withDataset,
} from './imagery';

// A rectangle over the reservoir's north shore, about 427 km² (the design's Reservoir-north is 412).
const reservoirNorth: DrawnRegion = {
  id: 'region-1',
  name: 'Reservoir-north',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[33.27, 46.87], [33.55, 46.87], [33.55, 46.69], [33.27, 46.69], [33.27, 46.87]]] },
  },
  bbox: [33.27, 46.69, 33.55, 46.87],
  createdAt: '2026-10-23T10:00:00.000Z',
};

describe('the imagery panel as a region', () => {
  it('starts on HLS L30 with the sample dates, ≤20 % cloud and B2–B6', () => {
    expect(defaultImagery(['2023-06-01', '2023-06-09'])).toEqual({
      dataset: 'HLS L30', start_date: '2023-06-01', end_date: '2023-06-09', cloud_max: 20, bands: ['B2', 'B3', 'B4', 'B5', 'B6'],
    });
    // Without the sample: the last 30 days.
    expect(defaultImagery(undefined, new Date(2026, 9, 23))).toMatchObject({ start_date: '2026-09-23', end_date: '2026-10-23' });
  });

  it('dataset, dates, cloud and bands become the ChatRegion that Send to chat puts in the composer', () => {
    let imagery = withDataset(defaultImagery(['2023-06-01', '2023-06-09']), 'Sentinel-2');
    expect(imagery.bands).toEqual(['B2', 'B3', 'B4', 'B8']);
    imagery = withDataset(imagery, 'HLS L30');
    imagery = toggleBand(toggleBand(imagery, 'B7'), 'B2');  // B7 on, B2 off
    imagery = toggleBand(imagery, 'B2');  // back on, in band order
    imagery = { ...imagery, cloud_max: 35 };

    const region = regionFromImagery(reservoirNorth, imagery);
    expect(region).toEqual({
      id: 'region-1',
      geometry: reservoirNorth.geometry.geometry,
      name: 'Reservoir-north',
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-09',
      cloud_max: 35,
      bands: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
      area_km2: expect.closeTo(426.7, 1),
    });
    // The chip and the request body (JobRegion forbids extra fields: no id, area or line).
    expect(regionLine(region)).toBe('Region · Reservoir-north · 427 km² · HLS L30 · 2023-06-01 → 06-09 · ≤35 % cloud');
    expect(toChatRegion(region)).toEqual({
      geometry: reservoirNorth.geometry.geometry,
      name: 'Reservoir-north',
      dataset: 'HLS L30',
      start_date: '2023-06-01',
      end_date: '2023-06-09',
      cloud_max: 35,
      bands: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
    });
  });

  it('the region card shows the centroid latitude first', () => {
    expect(formatCentroid(reservoirNorth)).toBe('46.78 N 33.41 E');
  });

  it('says why a region cannot be sent yet', () => {
    const imagery = defaultImagery(['2023-06-01', '2023-06-09']);
    expect(regionProblem(reservoirNorth, imagery)).toBeNull();
    expect(regionProblem(null, imagery)).toBe('Draw or select a region first');
    expect(regionProblem(reservoirNorth, { ...imagery, start_date: '2023-06-10' })).toBe('The start date is after the end date');
    expect(regionProblem(reservoirNorth, { ...imagery, bands: [] })).toBe('Pick at least one band');
    const huge = { ...reservoirNorth, geometry: { ...reservoirNorth.geometry, geometry: { type: 'Polygon' as const, coordinates: [[[30, 45], [32, 45], [32, 47], [30, 47], [30, 45]]] } } };
    expect(regionProblem(huge, imagery)).toBe('Regions are limited to 2 500 km²; draw a smaller one');
  });
});

describe('Fetch to map: disabled without Earth Engine, from GET /v1/capabilities', () => {
  const caps = (status: string) =>
    ({ tiers: [{}, {}, {}, {}, { id: 'data', items: [{ id: 'uploads', status: 'always' }, { id: 'earth-engine', status }] }, {}] }) as unknown as Capabilities;

  it('is enabled only when the data tier says earth-engine is configured', () => {
    expect(earthEngineStatus(caps('configured'), { pending: false, error: false })).toEqual({ configured: true, reason: null });
    expect(earthEngineStatus(caps('not_configured'), { pending: false, error: false })).toEqual({
      configured: false,
      reason: "Earth Engine isn't configured on this server",
    });
  });

  it('never enables while the capabilities are loading or failed, and says so', () => {
    expect(earthEngineStatus(undefined, { pending: true, error: false })).toEqual({
      configured: false,
      reason: 'Checking whether Earth Engine is configured…',
    });
    expect(earthEngineStatus(undefined, { pending: false, error: true })).toEqual({
      configured: false,
      reason: "Couldn't read this server's capabilities",
    });
  });
});

describe('Fetch to map: the bulk-fetch request', () => {
  it('fetches before and after windows around the dates, as a region job does', () => {
    // 8 days apart: ±3 days (gee_region.change_windows).
    expect(fetchPeriods('2023-06-01', '2023-06-09')).toEqual({ pre: ['2023-05-29', '2023-06-04'], post: ['2023-06-06', '2023-06-12'] });
    // Long ranges stop at ±8 days; one day apart or the same day: the days themselves.
    expect(fetchPeriods('2023-05-01', '2023-07-01')).toEqual({ pre: ['2023-04-23', '2023-05-09'], post: ['2023-06-23', '2023-07-09'] });
    expect(fetchPeriods('2023-06-01', '2023-06-02')).toEqual({ pre: ['2023-06-01', '2023-06-01'], post: ['2023-06-02', '2023-06-02'] });
  });

  it("names Landsat bands as the fetcher does (SR_B5) and keeps the region's bounds and cloud limit", () => {
    const imagery = { ...defaultImagery(['2023-06-01', '2023-06-09']), cloud_max: 20 };
    expect(bulkFetchRequest(reservoirNorth, imagery)).toEqual({
      bbox: [33.27, 46.69, 33.55, 46.87],
      dataset_id: 'landsat-8',
      bands: ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
      indices: [],
      pre_start: '2023-05-29',
      pre_end: '2023-06-04',
      post_start: '2023-06-06',
      post_end: '2023-06-12',
      cloud_cover_max: 20,
      use_now: false,
    });
    expect(bulkFetchRequest(reservoirNorth, withDataset(imagery, 'Sentinel-2'))).toMatchObject({ dataset_id: 'sentinel-2', bands: ['B2', 'B3', 'B4', 'B8'] });
  });
});
