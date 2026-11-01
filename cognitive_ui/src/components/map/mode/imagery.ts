import { area, centroid } from '@turf/turf';
import type { Capabilities } from '../../../api/hooks/useCapabilities';
import type { BulkFetchRequest } from '../../../api/hooks/useGEEData';
import type { DrawnRegion, Imagery } from '../../../store/useAppStore';
import type { ChatRegion, RegionDataset } from '../../../types';
import { REGION_DEFAULTS } from '../../chat/regionChip';

// Map mode's imagery panel: dataset, dates, cloud and
// bands become the region of POST /v1/jobs and /v1/chat (docs/API.md "Regions"), or
// an Earth Engine fetch through /v1/gee/bulk-fetch. Pure: the panel renders these.

export interface ImageryBand {
  code: string;  // the region's band code (server/schemas.py JobRegion.bands)
  label: string;  // the chip: "B2 Blue"
}

export interface ImageryDataset {
  id: RegionDataset;
  label: string;  // the segmented control
  description: string;  // the line under it
  bands: ImageryBand[];
  defaultBands: string[];
  /** The bulk-fetch dataset and its band names (dta/dti/data_sources/gee_region.py). */
  gee: { id: 'landsat-8' | 'sentinel-2' | 'modis'; prefix: string };
}

const band = (code: string, name: string, short = code): ImageryBand => ({ code, label: `${short} ${name}` });

export const IMAGERY_DATASETS: ImageryDataset[] = [
  {
    id: 'HLS L30',
    label: 'HLS L30',
    description: 'NASA HLS L30 (Landsat 8/9) · 30 m · revisit 8 days',
    bands: [band('B2', 'Blue'), band('B3', 'Green'), band('B4', 'Red'), band('B5', 'NIR'), band('B6', 'SWIR1'), band('B7', 'SWIR2')],
    defaultBands: ['B2', 'B3', 'B4', 'B5', 'B6'],
    gee: { id: 'landsat-8', prefix: 'SR_' },
  },
  {
    id: 'Sentinel-2',
    label: 'Sentinel-2',
    description: 'Copernicus Sentinel-2 L2A · 10 m · revisit 5 days',
    bands: [band('B2', 'Blue'), band('B3', 'Green'), band('B4', 'Red'), band('B8', 'NIR'), band('B11', 'SWIR1'), band('B12', 'SWIR2')],
    defaultBands: ['B2', 'B3', 'B4', 'B8'],
    gee: { id: 'sentinel-2', prefix: '' },
  },
  {
    id: 'MODIS',
    label: 'MODIS',
    description: 'NASA MODIS surface reflectance · 500 m · 8-day composite',
    bands: [
      band('sur_refl_b01', 'Red', 'b01'),
      band('sur_refl_b02', 'NIR', 'b02'),
      band('sur_refl_b03', 'Blue', 'b03'),
      band('sur_refl_b04', 'Green', 'b04'),
      band('sur_refl_b06', 'SWIR1', 'b06'),
      band('sur_refl_b07', 'SWIR2', 'b07'),
    ],
    defaultBands: ['sur_refl_b01', 'sur_refl_b02'],
    gee: { id: 'modis', prefix: '' },
  },
];

export function datasetOf(id: RegionDataset): ImageryDataset {
  return IMAGERY_DATASETS.find((d) => d.id === id) ?? IMAGERY_DATASETS[0];
}

function isoDay(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** The panel before any edit: HLS L30, the sample's dates (else the last 30 days), ≤20 % cloud. */
export function defaultImagery(dates?: [string, string], today = new Date()): Imagery {
  const start = new Date(today);
  start.setDate(today.getDate() - REGION_DEFAULTS.days);
  const [start_date, end_date] = dates ?? [isoDay(start), isoDay(today)];
  const dataset = datasetOf(REGION_DEFAULTS.dataset);
  return { dataset: dataset.id, start_date, end_date, cloud_max: REGION_DEFAULTS.cloud_max, bands: [...dataset.defaultBands] };
}

/** Another dataset: its default bands, the same dates and cloud limit. */
export function withDataset(imagery: Imagery, id: RegionDataset): Imagery {
  return { ...imagery, dataset: id, bands: [...datasetOf(id).defaultBands] };
}

/** A band chip clicked: on or off, kept in the dataset's order. */
export function toggleBand(imagery: Imagery, code: string): Imagery {
  const on = new Set(imagery.bands);
  if (on.has(code)) on.delete(code);
  else on.add(code);
  return { ...imagery, bands: datasetOf(imagery.dataset).bands.map((b) => b.code).filter((c) => on.has(c)) };
}

export const regionAreaKm2 = (region: DrawnRegion): number => area(region.geometry) / 1e6;

/** "46.78 N 33.41 E": the region's centroid, latitude first. */
export function formatCentroid(region: DrawnRegion): string {
  const [lon, lat] = centroid(region.geometry).geometry.coordinates;
  return `${Math.abs(lat).toFixed(2)} ${lat < 0 ? 'S' : 'N'} ${Math.abs(lon).toFixed(2)} ${lon < 0 ? 'W' : 'E'}`;
}

/** The region Send to chat puts in the composer (`setPendingRegion`), as the composer's From map does. */
export function regionFromImagery(region: DrawnRegion, imagery: Imagery): ChatRegion {
  return {
    id: region.id,
    geometry: region.geometry.geometry,
    ...(region.name ? { name: region.name } : {}),
    dataset: imagery.dataset,
    start_date: imagery.start_date,
    end_date: imagery.end_date,
    cloud_max: imagery.cloud_max,
    bands: [...imagery.bands],
    area_km2: regionAreaKm2(region),
  };
}

/** The server's limit on a region (server/regions.py MAX_REGION_KM2). */
export const MAX_REGION_KM2 = 2500;

/** Why the region can't be sent or fetched yet, or null when it can. */
export function regionProblem(region: DrawnRegion | null, imagery: Imagery): string | null {
  if (!region) return 'Draw or select a region first';
  if (imagery.start_date > imagery.end_date) return 'The start date is after the end date';
  if (imagery.bands.length === 0) return 'Pick at least one band';
  if (regionAreaKm2(region) > MAX_REGION_KM2) return 'Regions are limited to 2 500 km²; draw a smaller one';
  return null;
}

/** Whether this server can fetch from Earth Engine: `GET /v1/capabilities`, data tier, `earth-engine`. */
export function earthEngineStatus(
  capabilities: Capabilities | undefined,
  query: { pending: boolean; error: boolean },
): { configured: boolean; reason: string | null } {
  if (query.pending) return { configured: false, reason: 'Checking whether Earth Engine is configured…' };
  if (query.error || !capabilities) return { configured: false, reason: "Couldn't read this server's capabilities" };
  const item = capabilities.tiers[4].items.find((i) => i.id === 'earth-engine');
  if (item?.status === 'configured') return { configured: true, reason: null };
  return { configured: false, reason: "Earth Engine isn't configured on this server" };
}

const shiftDays = (day: string, days: number): string => {
  const d = new Date(`${day}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
};

/**
 * Before and after windows for a fetch, ±N days around the start and the end, as a
 * region job fetches them (dta/dti/data_sources/gee_region.py change_windows, N ≤ 8).
 */
export function fetchPeriods(start: string, end: string): { pre: [string, string]; post: [string, string] } {
  const days = Math.round((Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86_400_000);
  const half = Math.min(8, Math.floor(Math.max(days - 1, 0) / 2));
  return {
    pre: [shiftDays(start, -half), shiftDays(start, half)],
    post: [shiftDays(end, -half), shiftDays(end, half)],
  };
}

/** Fetch to map as `POST /v1/gee/bulk-fetch`: the region's bounds, the dataset's band names, both windows. */
export function bulkFetchRequest(region: DrawnRegion, imagery: Imagery): BulkFetchRequest {
  const { gee } = datasetOf(imagery.dataset);
  const { pre, post } = fetchPeriods(imagery.start_date, imagery.end_date);
  return {
    bbox: region.bbox,
    dataset_id: gee.id,
    bands: imagery.bands.map((b) => `${gee.prefix}${b}`),
    indices: [],
    pre_start: pre[0],
    pre_end: pre[1],
    post_start: post[0],
    post_end: post[1],
    cloud_cover_max: imagery.cloud_max,
    use_now: false,
  };
}
