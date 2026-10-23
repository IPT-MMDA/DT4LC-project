import type { JobRecord } from '../../../api/jobs';
import type { LayerLegend, LngLatBounds } from '../../../store/useAppStore';
import type { ChatRegion, MessageAttachment } from '../../../types';

// What a finished analysis shows as its result card (design/DESIGN.md §4), read from
// GET /v1/jobs/{id} `result.execution.artifacts` (docs/ARCHITECTURE.md "Job
// artifacts"). Pure: the card components render a ResultCardModel and nothing else.
// Images and downloads are artifact `url`s only; a pre-v0.5.0 base64 string is no
// reference and is ignored.

/** An artifact reference: `{kind: "geotiff" | "png", path, url, stats}`. */
export interface ArtifactRef {
  kind: string;
  path: string;
  url: string;
  stats?: Record<string, number | null>;
}

export type Tone = 'gained' | 'lost' | 'neutral';

export interface Stat {
  label: string;
  value: string;
  tone?: Tone;
}

export interface LegendItem {
  label: string;
  tone: Tone;
}

/** One image of a card: the map shown, the raster behind it and the overlay label. */
export interface CardView {
  key: 'difference' | 'before' | 'after' | 'map';
  label: string;  // the toggle's text
  layerName: string;  // its row in the Layers panel
  overlay: string;  // the mono label on the image, '' for none
  image?: ArtifactRef;
  raster?: ArtifactRef;
  legend?: LayerLegend;
}

export interface ClassItem {
  name: string;
  colour: string;
  value: string;  // per-class share, e.g. "38.5 %"
}

export interface BandRow {
  band: string;
  min: string;
  max: string;
  mean: string;
  std: string;
  valid: string;
}

interface CardBase {
  title: string;
  /** Composed (band math): the formula, dates and resolution, in mono under the title. */
  formula?: string;
  /** A mono note beside the title, e.g. "Delineate-Anything". */
  note?: string;
  source: string | null;
  bounds?: LngLatBounds;
  gpkgUrl?: string;
}

export type ResultCardModel =
  | (CardBase & { kind: 'index'; views: [CardView]; ramp?: { stops: string[]; range: [number, number] }; stats: Stat[] })
  | (CardBase & { kind: 'change'; views: CardView[]; legend: LegendItem[]; stats: Stat[] })
  | (CardBase & { kind: 'classification'; views: [CardView]; classes: ClassItem[] })
  | (CardBase & { kind: 'fields'; views: [CardView]; stats: Stat[] })
  | (CardBase & { kind: 'statistics'; views: []; bands: BandRow[] })
  | (CardBase & { kind: 'image'; views: [CardView] });

/** The files (or region) the analysis was asked about: the user message's context. */
export interface AnalysedContext {
  attachments: MessageAttachment[];
  region?: ChatRegion;
}

// --- Display tables, mirroring the server's renderers --------------------------

// dta/dti/algorithms/spectral_index.py COLORMAPS, as hex; band math renders with "ndwi".
const RAMPS: Record<string, string[]> = {
  ndvi: ['#994c1a', '#cc9933', '#ffff66', '#99cc33', '#339933', '#006600'],
  ndwi: ['#996633', '#ccb280', '#f2f2f2', '#99ccff', '#3380e6', '#003399'],
  ndsi: ['#664c33', '#b29966', '#ffe6b2', '#cce6ff', '#66b2ff', '#ffffff'],
};

// dta/registry.yaml `config.colormap`, `vmin`, `vmax` of each index algorithm.
const INDEX_DISPLAY: Record<string, { ramp: string; range: [number, number] }> = {
  ndvi: { ramp: 'ndvi', range: [-0.2, 0.8] },
  evi: { ramp: 'ndvi', range: [-0.2, 0.8] },
  ndwi: { ramp: 'ndwi', range: [-0.5, 0.5] },
  ndsi: { ramp: 'ndsi', range: [-0.5, 1.0] },
};

const INDEX_ARTIFACTS = ['NDVIMap', 'EVIMap', 'NDWIMap', 'NDSIMap'];

// dta/dti/algorithms/lulc_classifier.py LULC_CLASSES colours, by class_id.
const LULC_COLOURS: Record<number, string> = {
  1: '#0077be',
  2: '#ffffff',
  3: '#c2b280',
  4: '#d4e157',
  5: '#8bc34a',
  6: '#2e7d32',
};

// dta/dti/algorithms/snow_classifier.py snow classification colours.
const SNOW_COLOURS = { snow: '#ffcc66', other: '#66804c' };

/** Sensor (upload `metadata.sensor`, region `dataset`) → the source line's long name. */
export const SENSOR_NAMES: Record<string, string> = {
  'HLS L30': 'NASA HLS L30 (Landsat 8/9)',
  'HLS S30': 'NASA HLS S30 (Sentinel-2)',
  'Sentinel-2': 'Copernicus Sentinel-2',
  'Landsat 8/9': 'USGS Landsat 8/9',
  MODIS: 'NASA MODIS',
};

/** Change wording per index: the cover reported in `statistics.<key>` and the legend words. */
const CHANGE_WORDS: Record<string, { subject: string; cover?: { key: string; label: string }; gained: string; lost: string }> = {
  ndwi: { subject: 'Water', cover: { key: 'water', label: 'Open water' }, gained: 'water gained', lost: 'water lost' },
  ndvi: { subject: 'Vegetation', cover: { key: 'vegetation', label: 'Vegetation' }, gained: 'vegetation gained', lost: 'vegetation lost' },
  ndmi: { subject: 'Moisture', gained: 'wetter', lost: 'drier' },
};

// --- Formatting ------------------------------------------------------------------

const MINUS = '−';

/** Thousands grouped with spaces, as the design's statistics: 589824 → "589 824". */
function grouped(n: number): string {
  return Math.round(Math.abs(n)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * An area: one decimal under 100 km², whole km² from 100, "—" when unknown (a CRS
 * that is not projected). `sign` puts "+" before a positive value.
 */
export function formatKm2(value: unknown, { sign = false } = {}): string {
  if (!isNum(value)) return '—';
  const abs = Math.abs(value);
  const digits = abs < 100 ? abs.toFixed(1) : grouped(abs);
  const prefix = value < 0 ? MINUS : sign && value > 0 ? '+' : '';
  return `${prefix}${digits} km²`;
}

/** A statistic: integers grouped, small values with two decimals, a real minus sign. */
export function formatValue(value: unknown): string {
  if (!isNum(value)) return '—';
  const abs = Math.abs(value);
  const digits = Number.isInteger(value) || abs >= 1000 ? grouped(abs) : abs < 10 ? abs.toFixed(2) : abs.toFixed(1);
  return `${value < 0 ? MINUS : ''}${digits}`;
}

/** An end of a legend range as written: −0.5, 0.8. */
export const formatRangeEnd = (v: number): string => (v < 0 ? `${MINUS}${Math.abs(v)}` : String(v));

/** A share: "38.5 %", "<0.1 %" for a trace. */
export function formatPercent(value: unknown): string {
  if (!isNum(value)) return '—';
  if (value > 0 && value < 0.1) return '<0.1 %';
  return `${value.toFixed(1)} %`;
}

/** "(B5 - B6) / (B5 + B6)" → "(B5 − B6)/(B5 + B6)", as the design sets formulas. */
export function prettyFormula(formula: string): string {
  return formula.replace(/\s*\)\s*\/\s*\(\s*/g, ')/(').replace(/ - /g, ` ${MINUS} `);
}

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

/** "2023-06-09" → "9 June". */
export function dayMonth(date: string | undefined): string {
  const m = date ? /^\d{4}-(\d{2})-(\d{2})/.exec(date) : null;
  return m ? `${Number(m[2])} ${MONTHS[Number(m[1]) - 1]}` : '';
}

/** "2023-06-09" → "06-09". */
const monthDay = (date: string | undefined) => (date ? date.slice(5, 10) : '');

/** "06-01 → 06-09", or '' without both dates. */
function dateRange(before: string | undefined, after: string | undefined): string {
  return before && after ? `${monthDay(before)} → ${monthDay(after)}` : '';
}

function resolution(attachments: MessageAttachment[]): string {
  const r = attachments.map((a) => a.metadata?.resolution_m).find(isNum);
  if (r === undefined) return '';
  return `${Number.isInteger(r) ? r : r.toFixed(1)} m`;
}

/**
 * The card's source line from the analysed files' metadata, e.g. "NASA HLS L30
 * (Landsat 8/9) · 30 m"; a region's dataset when there are no files; null when
 * nothing is known.
 */
export function sourceLine(context: AnalysedContext): string | null {
  const sensors = context.attachments.map((a) => a.metadata?.sensor).filter((s): s is string => Boolean(s));
  if (sensors.length === 0 && context.region) sensors.push(context.region.dataset);
  const names = [...new Set(sensors)].map((s) => SENSOR_NAMES[s] ?? s);
  const parts = [names.join(' + '), resolution(context.attachments)].filter(Boolean);
  return parts.length > 0 ? parts.join(' · ') : null;
}

// --- Reading the result ---------------------------------------------------------

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Obj) : {});

/** `v` as an artifact reference, or undefined (null, a base64 string, anything else). */
export function asRef(v: unknown): ArtifactRef | undefined {
  const r = obj(v);
  return typeof r.url === 'string' && r.url && typeof r.path === 'string' ? (r as unknown as ArtifactRef) : undefined;
}

const firstRef = (group: unknown, kind: string): ArtifactRef | undefined =>
  Object.values(obj(group)).map(asRef).find((r) => r?.kind === kind);

/** The attachments that are the job's inputs (matched by path), or all of them. */
function analysed(context: AnalysedContext, paths: unknown[]): MessageAttachment[] {
  const wanted = paths.filter((p): p is string => typeof p === 'string');
  const hit = context.attachments.filter((a) => a.path && wanted.includes(a.path));
  return hit.length > 0 ? hit : context.attachments;
}

const byPath = (context: AnalysedContext, p: unknown) =>
  typeof p === 'string' ? context.attachments.find((a) => a.path === p) : undefined;

/** The union of the files' WGS84 bounds: what Fit to result zooms to. */
function boundsOf(files: MessageAttachment[]): LngLatBounds | undefined {
  const all = files.map((a) => a.metadata?.bounds_wgs84).filter((b): b is LngLatBounds => Array.isArray(b) && b.length === 4);
  if (all.length === 0) return undefined;
  return [
    Math.min(...all.map((b) => b[0])),
    Math.min(...all.map((b) => b[1])),
    Math.max(...all.map((b) => b[2])),
    Math.max(...all.map((b) => b[3])),
  ];
}

/** A GeoPackage download: a `gpkg` reference, or a model's `output_path` ending in .gpkg. */
function gpkgUrlOf(artifact: Obj): string | undefined {
  const refs = [...Object.values(obj(artifact.rasters)), ...Object.values(obj(artifact.vectors)), artifact.gpkg].map(asRef);
  const ref = refs.find((r) => r && (r.kind === 'gpkg' || r.path.endsWith('.gpkg')));
  if (ref) return ref.url;
  const out = artifact.output_path;
  return typeof out === 'string' && out.toLowerCase().endsWith('.gpkg') ? `/v1/download?path=${encodeURIComponent(out)}` : undefined;
}

const titled = (...parts: string[]) => parts.filter(Boolean).join(' · ');

function rampOf(index: string, displayRange?: unknown): { stops: string[]; range: [number, number] } | undefined {
  const d = INDEX_DISPLAY[index];
  const r = Array.isArray(displayRange) && displayRange.length === 2 && displayRange.every(isNum) ? (displayRange as [number, number]) : d?.range;
  const stops = RAMPS[d?.ramp ?? 'ndwi'];
  return r ? { stops, range: r } : undefined;
}

function indexCard(artifact: Obj, artifacts: Obj, context: AnalysedContext, bandMath: boolean): ResultCardModel {
  const index = String(artifact.index_type ?? '').toLowerCase();
  const name = bandMath ? String(artifact.name ?? artifact.formula ?? 'Band math') : index.toUpperCase();
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const file = byPath(context, artifacts.RasterPath) ?? byPath(context, artifact.path) ?? (files.length === 1 ? files[0] : undefined);
  const date = file?.metadata?.date;
  const stats = obj(artifact.statistics);
  const ramp = rampOf(bandMath ? 'band-math' : index, stats.display_range);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const image = asRef(vis[`${index}_map`]) ?? asRef(vis.index_map) ?? firstRef(vis, 'png');
  const raster = asRef(rasters[index]) ?? asRef(rasters.index) ?? firstRef(rasters, 'geotiff');
  const title = titled(name, dayMonth(date));
  return {
    kind: 'index',
    title,
    formula: bandMath && artifact.formula ? titled(prettyFormula(String(artifact.formula)), date ?? '', resolution(files)) : undefined,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    ramp,
    views: [{ key: 'map', label: 'Map', layerName: title, overlay: '', image, raster, legend: ramp && { kind: 'ramp', ...ramp } }],
    stats: [
      { label: 'Mean', value: formatValue(stats.mean) },
      { label: 'Min', value: formatValue(stats.min) },
      { label: 'Max', value: formatValue(stats.max) },
      { label: 'Valid pixels', value: formatValue(stats.valid_pixels) },
    ],
  };
}

function changeCard(artifact: Obj, artifacts: Obj, context: AnalysedContext, bandMath: boolean): ResultCardModel {
  const index = bandMath ? String(artifact.name ?? '') : String(artifact.index_type ?? '');
  const key = index.toLowerCase();
  const label = bandMath ? index || 'Band math' : String(artifact.index_name ?? index.toUpperCase());
  const words = CHANGE_WORDS[key];
  const files = analysed(context, [artifacts.RasterPathBefore, artifacts.RasterPathAfter]);
  const before = byPath(context, artifacts.RasterPathBefore)?.metadata?.date ?? files[0]?.metadata?.date;
  const after = byPath(context, artifacts.RasterPathAfter)?.metadata?.date ?? files[files.length - 1]?.metadata?.date;
  const stats = obj(artifact.statistics);
  const areas = obj(stats.areas_km2);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const title = bandMath ? (words ? `${words.subject} change · Δ${label}` : `${label} change`) : words ? `${words.subject} change · ${label}` : `${label} change`;
  const gained = words?.gained ?? 'increase';
  const lost = words?.lost ?? 'decrease';
  const range = dateRange(before, after);

  let statRow: Stat[];
  const cover = words?.cover ? obj(stats[words.cover.key]) : {};
  if (words?.cover && Object.keys(cover).length > 0) {
    statRow = [
      { label: `${words.cover.label} before`, value: formatKm2(cover.before_km2) },
      { label: `${words.cover.label} after`, value: formatKm2(cover.after_km2) },
      { label: 'Gained', value: formatKm2(cover.gained_km2, { sign: true }), tone: 'gained' },
      { label: 'Lost', value: formatKm2(cover.lost_km2), tone: 'lost' },
    ];
  } else {
    const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
    statRow = [
      { label: cap(lost), value: formatKm2(areas.decrease ?? stats.decrease_area_km2), tone: 'lost' },
      { label: cap(gained), value: formatKm2(areas.increase ?? stats.increase_area_km2, { sign: true }), tone: 'gained' },
    ];
    if (bandMath && isNum(stats.threshold)) statRow.push({ label: 'Threshold', value: `Δ${label} ${stats.threshold}` });
  }

  const legend: LegendItem[] = bandMath
    ? [{ label: lost, tone: 'lost' }, { label: gained, tone: 'gained' }, { label: 'no change', tone: 'neutral' }]
    : [{ label: gained, tone: 'gained' }, { label: lost, tone: 'lost' }, { label: 'no change', tone: 'neutral' }];
  const indexRamp = rampOf(bandMath ? 'band-math' : key);
  const indexLegend: LayerLegend | undefined = indexRamp && { kind: 'ramp', ...indexRamp };
  const shortName = bandMath ? label : String(artifact.index_name ?? label);

  return {
    kind: 'change',
    title,
    formula: bandMath && artifact.formula ? titled(prettyFormula(String(artifact.formula)), range, resolution(files)) : undefined,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    legend,
    stats: statRow,
    views: [
      {
        key: 'difference',
        label: 'Difference',
        layerName: title,
        overlay: titled('difference', range),
        image: asRef(vis.change_map),
        raster: asRef(rasters.change),
        legend: { kind: 'change' },
      },
      {
        key: 'before',
        label: 'Before',
        layerName: titled(shortName, before ?? 'before'),
        overlay: titled(`${shortName} before`, before ?? ''),
        image: asRef(vis.index_before) ?? asRef(vis[`${key}_before`]),
        raster: asRef(rasters.index_before),
        legend: indexLegend,
      },
      {
        key: 'after',
        label: 'After',
        layerName: titled(shortName, after ?? 'after'),
        overlay: titled(`${shortName} after`, after ?? ''),
        image: asRef(vis.index_after) ?? asRef(vis[`${key}_after`]),
        raster: asRef(rasters.index_after),
        legend: indexLegend,
      },
    ],
  };
}

function singleFile(context: AnalysedContext, artifacts: Obj, artifact: Obj) {
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const file = byPath(context, artifacts.RasterPath) ?? (files.length === 1 ? files[0] : undefined);
  return { files, date: file?.metadata?.date, file };
}

function lulcCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date } = singleFile(context, artifacts, artifact);
  const title = titled('Land cover', dayMonth(date));
  const classes = Object.entries(obj(artifact.class_statistics))
    .map(([name, v]) => ({ name, ...(obj(v) as { class_id?: number; pixels?: number; percentage?: number }) }))
    .filter((c) => (c.pixels ?? 0) > 0)
    .map((c) => ({ name: c.name, colour: LULC_COLOURS[c.class_id ?? 0] ?? '#000000', value: formatPercent(c.percentage) }));
  return {
    kind: 'classification',
    title,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    classes,
    views: [{
      key: 'map', label: 'Map', layerName: title, overlay: '',
      image: asRef(obj(artifact.visualizations).classification_map),
      raster: asRef(obj(artifact.rasters).classification),
    }],
  };
}

function snowCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date } = singleFile(context, artifacts, artifact);
  const title = titled('Snow and ice', dayMonth(date));
  const share = obj(artifact.statistics).snow_coverage_percent;
  return {
    kind: 'classification',
    title,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    classes: [
      { name: 'Snow / ice', colour: SNOW_COLOURS.snow, value: formatPercent(share) },
      { name: 'Other', colour: SNOW_COLOURS.other, value: isNum(share) ? formatPercent(100 - share) : '—' },
    ],
    views: [{
      key: 'map', label: 'Map', layerName: title, overlay: '',
      image: asRef(obj(artifact.visualizations).snow_classification),
      raster: asRef(obj(artifact.rasters).snow_mask),
    }],
  };
}

function fieldsCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date } = singleFile(context, artifacts, artifact);
  const title = titled('Field boundaries', dayMonth(date));
  const count = artifact.num_fields;
  const area = artifact.total_area_m2;
  const geographic = /EPSG:4326$/i.test(String(artifact.crs ?? ''));
  const stats: Stat[] = [{ label: 'Fields', value: formatValue(count) }];
  if (isNum(area) && !geographic) {
    stats.push({ label: 'Total area', value: formatKm2(area / 1e6) });
    if (isNum(count) && count > 0) stats.push({ label: 'Mean size', value: `${formatValue(area / count / 1e4)} ha` });
  }
  return {
    kind: 'fields',
    title,
    note: 'Delineate-Anything',
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    gpkgUrl: gpkgUrlOf(artifact),
    stats,
    views: [{ key: 'map', label: 'Map', layerName: title, overlay: '', image: asRef(obj(artifact.visualizations).field_boundaries) }],
  };
}

function statisticsCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date, file } = singleFile(context, artifacts, artifact);
  const codes = (file ?? files[0])?.metadata?.bands ?? [];
  const bands = Object.entries(obj(artifact.bands)).map(([key, v]) => {
    const n = Number(/(\d+)$/.exec(key)?.[1]);
    const s = obj(v);
    return {
      band: codes.find((b) => b.index === n)?.code ?? (Number.isFinite(n) ? `Band ${n}` : key),
      min: formatValue(s.min),
      max: formatValue(s.max),
      mean: formatValue(s.mean),
      std: formatValue(s.std),
      valid: formatValue(s.valid_pixels),
    };
  });
  return {
    kind: 'statistics',
    title: titled('Band statistics', dayMonth(date)),
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    bands,
    views: [],
  };
}

/** Any other step output with an image (a model's outputs): the image and its raster. */
function imageCard(artifact: Obj, context: AnalysedContext, title: string): ResultCardModel | null {
  const image = firstRef(artifact.visualizations, 'png');
  if (!image) return null;
  return {
    kind: 'image',
    title,
    source: sourceLine(context),
    bounds: boundsOf(context.attachments),
    gpkgUrl: gpkgUrlOf(artifact),
    views: [{ key: 'map', label: 'Map', layerName: title, overlay: '', image, raster: firstRef(artifact.rasters, 'geotiff') }],
  };
}

/**
 * The card for a finished job, picked by artifact kind: change (ChangeMap,
 * BandMathChangeMap), index (NDVI/EVI/NDWI/NDSI, BandMathMap), classification
 * (LULCMap, SnowClassification), field boundaries, statistics; else the first
 * image of any step output, titled `fallbackTitle`. Null when there is nothing to show.
 */
export function resultCardOf(job: Pick<JobRecord, 'result'>, context: AnalysedContext, fallbackTitle = 'Result'): ResultCardModel | null {
  const artifacts = obj(obj(obj(job.result).execution).artifacts);
  const get = (k: string) => obj(artifacts[k]);
  const has = (k: string) => Object.keys(get(k)).length > 0;

  if (has('BandMathChangeMap')) return changeCard(get('BandMathChangeMap'), artifacts, context, true);
  if (has('ChangeMap')) return changeCard(get('ChangeMap'), artifacts, context, false);
  if (has('BandMathMap')) return indexCard(get('BandMathMap'), artifacts, context, true);
  const index = INDEX_ARTIFACTS.find(has);
  if (index) return indexCard(get(index), artifacts, context, false);
  if (has('LULCMap')) return lulcCard(get('LULCMap'), artifacts, context);
  if (has('SnowClassification')) return snowCard(get('SnowClassification'), artifacts, context);
  if (has('FieldBoundaries')) return fieldsCard(get('FieldBoundaries'), artifacts, context);
  if (has('Statistics')) return statisticsCard(get('Statistics'), artifacts, context);
  for (const [key, value] of Object.entries(artifacts)) {
    if (key === 'agent_result') continue;
    const card = imageCard(obj(value), context, fallbackTitle);
    if (card) return card;
  }
  return null;
}
