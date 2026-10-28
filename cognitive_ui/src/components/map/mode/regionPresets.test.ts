import { describe, expect, it } from 'vitest';
import type { MessageAttachment } from '../../../types';
import { formatCentroid, regionAreaKm2 } from './imagery';
import { GeoJSONError, presetRegion, regionFromGeoJSON } from './regionPresets';

const NOW = new Date('2026-10-23T10:00:00.000Z');
// The sample's footprint (BR1 metadata.footprint_wgs84), NW NE SE SW.
const FOOTPRINT: [number, number][] = [[32.3888, 46.6634], [32.6901, 46.6639], [32.6911, 46.4567], [32.3906, 46.4553]];
const SAMPLE: MessageAttachment[] = [
  { id: 's', filename: 'hlsl_20230601.tif', path: '/u/s_hlsl_20230601.tif', metadata: { date: '2023-06-01' } },
  { id: 's', filename: 'hlsl_20230609.tif', path: '/u/s_hlsl_20230609.tif', metadata: { date: '2023-06-09', footprint_wgs84: FOOTPRINT } },
];

describe('region presets', () => {
  it("Kakhovka is the sample's footprint when a sample is known", () => {
    const region = presetRegion('kakhovka', SAMPLE, NOW);
    expect(region).toMatchObject({ id: 'preset-kakhovka', name: 'Kakhovka sample area', type: 'polygon', createdAt: NOW.toISOString() });
    expect(region.geometry.geometry).toEqual({ type: 'Polygon', coordinates: [[...FOOTPRINT, FOOTPRINT[0]]] });
    expect(region.bbox).toEqual([32.3888, 46.4553, 32.6911, 46.6639]);
    expect(formatCentroid(region)).toBe('46.56 N 32.54 E');
  });

  it('Kakhovka is a box around the dam without a sample', () => {
    const region = presetRegion('kakhovka', [], NOW);
    expect(region).toMatchObject({ id: 'preset-kakhovka', name: 'Kakhovka dam' });
    expect(formatCentroid(region)).toBe('46.78 N 33.37 E');
    expect(regionAreaKm2(region)).toBeGreaterThan(100);
    expect(regionAreaKm2(region)).toBeLessThan(2500);
  });

  it('Brienz is a box around Lake Brienz, whatever the sample', () => {
    const region = presetRegion('brienz', SAMPLE, NOW);
    expect(region).toMatchObject({ id: 'preset-brienz', name: 'Lake Brienz', type: 'polygon' });
    expect(formatCentroid(region)).toBe('46.73 N 7.97 E');
    expect(regionAreaKm2(region)).toBeLessThan(2500);
  });
});

const square = [[[7.9, 46.7], [8.0, 46.7], [8.0, 46.75], [7.9, 46.75], [7.9, 46.7]]];
const parse = (value: unknown, name = 'lake.geojson') => regionFromGeoJSON(typeof value === 'string' ? value : JSON.stringify(value), name, NOW);

describe('Upload GeoJSON', () => {
  it('accepts a Polygon Feature, named by its properties', () => {
    const region = parse({ type: 'Feature', properties: { name: 'Bönigen shore' }, geometry: { type: 'Polygon', coordinates: square } });
    expect(region).toMatchObject({ id: `geojson-${NOW.getTime()}`, name: 'Bönigen shore', type: 'polygon', bbox: [7.9, 46.7, 8.0, 46.75] });
    expect(region.geometry).toMatchObject({ type: 'Feature', geometry: { type: 'Polygon', coordinates: square } });
  });

  it('accepts a bare Polygon or MultiPolygon geometry, named by the file', () => {
    expect(parse({ type: 'Polygon', coordinates: square }, 'shore.geojson')).toMatchObject({ name: 'shore', geometry: { geometry: { type: 'Polygon' } } });
    expect(parse({ type: 'MultiPolygon', coordinates: [square, square] }, 'two.json').geometry.geometry).toMatchObject({ type: 'MultiPolygon' });
  });

  it('accepts a FeatureCollection: one polygon as it is, several as one MultiPolygon, other features skipped', () => {
    const one = parse({ type: 'FeatureCollection', features: [{ type: 'Feature', properties: { name: 'Only' }, geometry: { type: 'Polygon', coordinates: square } }] });
    expect(one).toMatchObject({ name: 'Only', geometry: { geometry: { type: 'Polygon' } } });
    const many = parse({
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [7.9, 46.7] } },
        { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: square } },
        { type: 'Feature', properties: {}, geometry: { type: 'MultiPolygon', coordinates: [square] } },
      ],
    });
    expect(many.geometry.geometry).toEqual({ type: 'MultiPolygon', coordinates: [square, square] });
    expect(many.name).toBe('lake');
  });

  it('closes an open ring', () => {
    const open = [[[7.9, 46.7], [8.0, 46.7], [8.0, 46.75]]];
    expect((parse({ type: 'Polygon', coordinates: open }).geometry.geometry as GeoJSON.Polygon).coordinates[0]).toEqual([[7.9, 46.7], [8.0, 46.7], [8.0, 46.75], [7.9, 46.7]]);
  });

  it.each([
    ['not JSON', '{"type": ', "lake.geojson isn't valid JSON."],
    ['JSON that is not GeoJSON', [1, 2], "lake.geojson isn't GeoJSON."],
    ['a point', { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [7.9, 46.7] } }, 'lake.geojson has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.'],
    ['a line', { type: 'LineString', coordinates: [[7.9, 46.7], [8, 46.7]] }, 'lake.geojson has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.'],
    ['an empty collection', { type: 'FeatureCollection', features: [] }, 'lake.geojson has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.'],
    ['projected coordinates', { type: 'Polygon', coordinates: [[[453240, 5144820], [476280, 5144820], [476280, 5167860], [453240, 5144820]]] }, "lake.geojson isn't in longitude/latitude (WGS84). Export it as EPSG:4326 GeoJSON."],
    ['a ring of two points', { type: 'Polygon', coordinates: [[[7.9, 46.7], [8, 46.7]]] }, 'lake.geojson has a polygon ring with fewer than 3 points.'],
    ['a polygon without coordinates', { type: 'Polygon', coordinates: [] }, 'lake.geojson has a polygon without coordinates.'],
    ['non-numeric coordinates', { type: 'Polygon', coordinates: [[['a', 'b'], [8, 46.7], [8, 46.75]]] }, 'lake.geojson has a polygon ring with fewer than 3 points.'],
  ])('refuses %s', (_, value, message) => {
    expect(() => parse(value)).toThrow(GeoJSONError);
    expect(() => parse(value)).toThrow(message);
  });
});
