import { bbox as turfBbox } from '@turf/turf';
import type { DrawnRegion } from '../../../store/useAppStore';
import type { MessageAttachment } from '../../../types';

// The presets under map mode's region card:
// Kakhovka dam, Brienz and Upload GeoJSON. Each becomes a drawn region in useAppStore.
// Pure: ImageryPanel adds and selects what these return.

export type PresetId = 'kakhovka' | 'brienz';

type Ring = [number, number][];

/** A closed lon/lat box. */
const box = (west: number, south: number, east: number, north: number): Ring => [
  [west, north], [east, north], [east, south], [west, south], [west, north],
];

// Around the Kakhovka dam (46.78 N 33.37 E) and Lake Brienz (46.73 N 7.97 E).
const KAKHOVKA_DAM = box(33.27, 46.71, 33.47, 46.85);
const LAKE_BRIENZ = box(7.85, 46.68, 8.09, 46.78);

function region(id: string, name: string, geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon, now: Date): DrawnRegion {
  const feature: GeoJSON.Feature = { type: 'Feature', properties: { name }, geometry };
  return {
    id,
    name,
    type: 'polygon',
    geometry: feature,
    bbox: turfBbox(feature) as DrawnRegion['bbox'],
    createdAt: now.toISOString(),
  };
}

/** The sample's footprint (`metadata.footprint_wgs84`, NW NE SE SW) as a closed ring, if a file has one. */
function sampleFootprint(sample: MessageAttachment[]): Ring | null {
  const corners = sample.map((a) => a.metadata?.footprint_wgs84).find((f) => Array.isArray(f) && f.length >= 3);
  return corners ? [...corners, corners[0]] : null;
}

/**
 * A preset as a drawn region with a stable id (`preset-<id>`), so picking it again
 * selects the same region. Kakhovka is the sample's footprint when a sample is known,
 * else a box around the dam; Brienz is a box around the lake (there is no Brienz sample).
 */
export function presetRegion(id: PresetId, sample: MessageAttachment[], now = new Date()): DrawnRegion {
  if (id === 'brienz') return region('preset-brienz', 'Lake Brienz', { type: 'Polygon', coordinates: [LAKE_BRIENZ] }, now);
  const footprint = sampleFootprint(sample);
  return footprint
    ? region('preset-kakhovka', 'Kakhovka sample area', { type: 'Polygon', coordinates: [footprint] }, now)
    : region('preset-kakhovka', 'Kakhovka dam', { type: 'Polygon', coordinates: [KAKHOVKA_DAM] }, now);
}

// --- Upload GeoJSON --------------------------------------------------------------------

/** Why a file can't be a region: the sentence the panel shows. */
export class GeoJSONError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GeoJSONError';
  }
}

type Polygonal = GeoJSON.Polygon | GeoJSON.MultiPolygon;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

function isPosition(p: unknown): p is [number, number] {
  return Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
}

/** A ring of at least four positions (closed or not; an open one is closed). */
function ringOf(raw: unknown, file: string): Ring {
  if (!Array.isArray(raw) || !raw.every(isPosition) || raw.length < 3) throw new GeoJSONError(`${file} has a polygon ring with fewer than 3 points.`);
  const ring = raw.map(([x, y]) => [x, y] as [number, number]);
  if (ring.some(([x, y]) => Math.abs(x) > 180 || Math.abs(y) > 90)) {
    throw new GeoJSONError(`${file} isn't in longitude/latitude (WGS84). Export it as EPSG:4326 GeoJSON.`);
  }
  const [first, last] = [ring[0], ring[ring.length - 1]];
  return first[0] === last[0] && first[1] === last[1] ? ring : [...ring, first];
}

const polygonOf = (raw: unknown, file: string): Ring[] => {
  if (!Array.isArray(raw) || raw.length === 0) throw new GeoJSONError(`${file} has a polygon without coordinates.`);
  return raw.map((r) => ringOf(r, file));
};

/** The Polygon or MultiPolygon geometries in a GeoJSON object; others are skipped. */
function polygonsIn(value: unknown, file: string): Polygonal[] {
  if (!isObject(value)) return [];
  switch (value.type) {
    case 'Polygon':
      return [{ type: 'Polygon', coordinates: polygonOf(value.coordinates, file) }];
    case 'MultiPolygon':
      if (!Array.isArray(value.coordinates) || value.coordinates.length === 0) throw new GeoJSONError(`${file} has a MultiPolygon without coordinates.`);
      return [{ type: 'MultiPolygon', coordinates: value.coordinates.map((p) => polygonOf(p, file)) }];
    case 'Feature':
      return polygonsIn(value.geometry, file);
    case 'FeatureCollection':
      return Array.isArray(value.features) ? value.features.flatMap((f) => polygonsIn(f, file)) : [];
    case 'GeometryCollection':
      return Array.isArray(value.geometries) ? value.geometries.flatMap((g) => polygonsIn(g, file)) : [];
    default:
      return [];
  }
}

/** A feature's or collection's `properties.name` / `name`, for the region's name. */
function nameIn(value: Record<string, unknown>): string | null {
  const props = isObject(value.properties) ? value.properties : {};
  const first = value.type === 'FeatureCollection' && Array.isArray(value.features) && isObject(value.features[0]) ? value.features[0] : null;
  const firstProps = first && isObject(first.properties) ? first.properties : {};
  const name = props.name ?? value.name ?? (Array.isArray(value.features) && value.features.length === 1 ? firstProps.name : undefined);
  return typeof name === 'string' && name.trim() ? name.trim().slice(0, 80) : null;
}

/**
 * An uploaded GeoJSON file as a drawn region: a Polygon or MultiPolygon Feature,
 * FeatureCollection or bare geometry (several polygons become one MultiPolygon).
 * Throws `GeoJSONError` with a sentence for anything else.
 */
export function regionFromGeoJSON(text: string, fileName: string, now = new Date()): DrawnRegion {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new GeoJSONError(`${fileName} isn't valid JSON.`);
  }
  if (!isObject(value) || typeof value.type !== 'string') throw new GeoJSONError(`${fileName} isn't GeoJSON.`);
  const polygons = polygonsIn(value, fileName);
  if (polygons.length === 0) throw new GeoJSONError(`${fileName} has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.`);
  const geometry: Polygonal =
    polygons.length === 1
      ? polygons[0]
      : { type: 'MultiPolygon', coordinates: polygons.flatMap((p) => (p.type === 'Polygon' ? [p.coordinates] : p.coordinates)) };
  const name = nameIn(value) ?? fileName.replace(/\.(geo)?json$/i, '');
  return region(`geojson-${now.getTime()}`, name, geometry, now);
}
