import type { Job } from '../types';

/** One raster the map can show (path is server path for `/v1/tiles/...?path=`). */
export interface GeoTIFFOutput {
  path: string;
  label: string;
  layerId: string;
}

function artifactsOf(result: Job['result'] | undefined): Record<string, unknown> | null {
  if (!result || typeof result !== 'object') return null;
  const r = result as Record<string, unknown>;
  const ex = r.execution as Record<string, unknown> | undefined;
  if (ex?.artifacts && typeof ex.artifacts === 'object' && !Array.isArray(ex.artifacts)) {
    return ex.artifacts as Record<string, unknown>;
  }
  if (r.artifacts && typeof r.artifacts === 'object' && !Array.isArray(r.artifacts)) {
    return r.artifacts as Record<string, unknown>;
  }
  return null;
}

function pathForArtifactKey(key: string, a: Record<string, unknown>): string | null {
  const out = a.output_path;
  if (typeof out === 'string' && out.length > 0) return out;
  const p = a.path;
  if (typeof p === 'string' && p.length > 0) return p;
  if (key === 'ChangeMap') {
    const meta = a.metadata as Record<string, unknown> | undefined;
    const ap = meta?.after_path ?? meta?.before_path;
    if (typeof ap === 'string' && ap.length > 0) return ap;
  }
  return null;
}

const MAP = { NDVIMap: 'NDVI', NDWIMap: 'NDWI', NDSIMap: 'NDSI', ChangeMap: 'Change', LULCMap: 'LULC' } as const;
const CAN_USE_RASTER_PATH = new Set(['NDVIMap', 'NDWIMap', 'NDSIMap', 'LULCMap']);

/** Paths to show for "View on Map" — one function, used by job page, chat, and map layers. */
export function getGeoTIFFOutputsFromJob(job: Job): GeoTIFFOutput[] {
  const artifacts = artifactsOf(job.result);
  if (!artifacts) return [];

  const inputTif = typeof artifacts.RasterPath === 'string' && artifacts.RasterPath.length > 0 ? artifacts.RasterPath : null;
  const jobId = job.id?.length ? job.id : 'job';
  const short = jobId.slice(0, 8);
  const out: GeoTIFFOutput[] = [];

  for (const [key, label] of Object.entries(MAP)) {
    const raw = artifacts[key];
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue;
    const a = raw as Record<string, unknown>;
    let path = pathForArtifactKey(key, a);
    if (!path && inputTif && CAN_USE_RASTER_PATH.has(key)) path = inputTif;
    if (path) {
      out.push({
        path,
        label: `${label} - ${short}`,
        layerId: `job-${jobId}-${key.toLowerCase().replace('map', '')}`,
      });
    }
  }

  if (out.length === 0 && inputTif) {
    out.push({ path: inputTif, label: `GeoTIFF - ${short}`, layerId: `job-${jobId}-geotiff` });
  }
  return out;
}

/** Same as `artifactsOf` but exported for NDVI/change UI sections that read `execution.artifacts` or flat `artifacts`. */
export function getArtifactsFromJobResult(result: Job['result'] | undefined): Record<string, unknown> | null {
  return artifactsOf(result);
}
