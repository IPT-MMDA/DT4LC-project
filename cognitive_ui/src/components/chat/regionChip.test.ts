import { describe, expect, it } from 'vitest';
import type { DrawnRegion } from '../../store/useAppStore';
import type { ChatRegion } from '../../types';
import { dateRange, formatArea, regionChipLines, regionFromMap, regionLine, sampleDates } from './regionChip';

const REGION: ChatRegion = {
  id: 'r1',
  geometry: { type: 'Polygon', coordinates: [] },
  name: 'Dnipro floodplain',
  place: 'Kherson',
  dataset: 'HLS L30',
  start_date: '2023-06-01',
  end_date: '2023-06-09',
  cloud_max: 20,
  area_km2: 412.3,
};

// A 0.1° × 0.1° box at the sample's latitude: about 85 km² (turf's spherical area).
const DRAWN: DrawnRegion = {
  id: 'region-1',
  name: 'Reservoir-north',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[32.5, 46.5], [32.6, 46.5], [32.6, 46.6], [32.5, 46.6], [32.5, 46.5]]] },
  },
  bbox: [32.5, 46.5, 32.6, 46.6],
  createdAt: '2026-10-23T00:00:00Z',
};

describe('region chip line (DESIGN §4)', () => {
  it('reads like the server line', () => {
    expect(regionLine(REGION)).toBe(
      'Region · Dnipro floodplain · Kherson · 412 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud',
    );
    expect(regionChipLines(REGION)).toEqual({
      title: 'Region · Dnipro floodplain · Kherson',
      detail: '412 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud',
    });
  });

  it("uses the server's echoed line once the job exists", () => {
    const echoed = { ...REGION, line: 'Region · Dnipro floodplain · Kherson · 537 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud' };
    expect(regionLine(echoed)).toBe(echoed.line);
    expect(regionChipLines(echoed).detail).toBe('537 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud');
  });

  it('leaves out a missing name and place, formats small areas and date ranges as the server does', () => {
    expect(regionLine({ ...REGION, name: undefined, place: undefined, area_km2: 8.04 })).toBe(
      'Region · 8.0 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud',
    );
    expect(formatArea(2499.6)).toBe('2500 km²');
    expect(dateRange('2022-12-20', '2023-01-19')).toBe('2022-12-20 → 2023-01-19');
    expect(dateRange('2023-06-01', '2023-06-01')).toBe('2023-06-01');
  });
});

describe('regionFromMap', () => {
  it('defaults to HLS L30, the last 30 days ending today and ≤20 % cloud, with the area from turf', () => {
    const region = regionFromMap(DRAWN, undefined, new Date(2026, 9, 23));
    expect(region).toMatchObject({
      id: 'region-1',
      name: 'Reservoir-north',
      dataset: 'HLS L30',
      start_date: '2026-09-23',
      end_date: '2026-10-23',
      cloud_max: 20,
      geometry: DRAWN.geometry.geometry,
    });
    expect(region.area_km2).toBeGreaterThan(80);
    expect(region.area_km2).toBeLessThan(90);
    expect(regionLine(region)).toMatch(/^Region · Reservoir-north · 8\d km² · HLS L30 · 2026-09-23 → 10-23 · ≤20 % cloud$/);
  });

  it("takes the sample's dates when the conversation has the sample", () => {
    const group = { name: 'kakhovka_hls_l30', line: '' };
    const dates = sampleDates([
      { id: 's', filename: 'b.tif', group, metadata: { date: '2023-06-09' } },
      { id: 's', filename: 'a.tif', group, metadata: { date: '2023-06-01' } },
      { id: 'o', filename: 'own.tif', metadata: { date: '2024-07-11' } },
    ]);
    expect(dates).toEqual(['2023-06-01', '2023-06-09']);
    expect(regionFromMap(DRAWN, dates)).toMatchObject({ start_date: '2023-06-01', end_date: '2023-06-09' });
    expect(sampleDates([])).toBeUndefined();
  });
});
