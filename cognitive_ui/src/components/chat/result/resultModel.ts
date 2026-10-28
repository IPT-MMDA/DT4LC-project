import type { JobRecord } from '../../../api/jobs';
import type { Layer, LayerLegend, LegendClass, LngLatBounds } from '../../../store/useAppStore';
import type { ChatRegion, MessageAttachment } from '../../../types';

// What a finished analysis shows as its result card, read from
// GET /v1/jobs/{id} `result.execution.artifacts` (docs/ARCHITECTURE.md "Job
// artifacts"). Pure: the card components render a ResultCardModel and nothing else.
// Images and downloads are artifact `url`s only; a pre-v0.5.0 base64 string is no
// reference and is ignored. Colours come from the server: a bare map's `stats.style`
// and `class_statistics[*].colour`, never from tables mirrored here.

/** An artifact reference: `{kind: "geotiff" | "png", path, url, stats}`. */
export interface ArtifactRef {
  kind: string;
  path: string;
  url: string;  // carries `&name=<download_name>` when the job has one
  stats?: Record<string, unknown>;
  download_name?: string;  // the readable file name; older jobs have none
}

/** The file name a download gets: the ref's `download_name`, else its stored name (older jobs). */
export function downloadName(ref: Pick<ArtifactRef, 'path' | 'download_name'>): string {
  return ref.download_name || ref.path.split('/').pop() || ref.path;
}

/** How a raster is drawn: a bare map PNG's `stats.style`, the same as its tiles. */
export type RasterStyle =
  | { kind: 'change' | 'index'; range: [number, number]; ramp: string[] }
  | { kind: 'class'; classes: Record<string, string> };

export type Tone = 'gained' | 'lost' | 'neutral';

export interface Stat {
  label: string;
  value: string;
  tone?: Tone;
}

export interface LegendItem {
  label: string;
  tone: Tone;
  colour?: string;  // from the change style's ramp; the legend-* token without one
}

/** One image of a card: the map shown, the raster behind it and the overlay label. */
export interface CardView {
  key: string;  // 'difference' | 'before' | 'after' | 'map', or a raster key of a land-cover card
  label: string;  // the toggle's text
  layerName: string;  // its row in the Layers panel
  overlay: string;  // the mono label on the image, '' for none
  image?: ArtifactRef;  // the full figure (title, colour bar): the PNG download
  bare?: ArtifactRef;  // the map alone, shown full-bleed; older jobs have none
  raster?: ArtifactRef;
  legend?: LayerLegend;
  /** A map drawn faded under `bare` (the start map under the changed cells). */
  base?: ArtifactRef;
  /** The view's own legend items (risk levels with their areas), when not the card's classes. */
  items?: ClassItem[];
}

export interface ClassItem {
  name: string;
  colour?: string;
  value: string;  // per-class area, e.g. "204 km²", or share, "38.5 %"
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
  gpkg?: Pick<ArtifactRef, 'path' | 'url' | 'download_name'>;
}

export type ResultCardModel =
  | (CardBase & { kind: 'index'; views: [CardView]; ramp?: { stops: string[]; range: [number, number] }; stats: Stat[] })
  | (CardBase & { kind: 'change'; views: CardView[]; legend: LegendItem[]; stats: Stat[] })
  | (CardBase & { kind: 'classification'; views: [CardView]; classes: ClassItem[] })
  | (CardBase & { kind: 'fields'; views: [CardView]; stats: Stat[] })
  | (CardBase & { kind: 'statistics'; views: []; bands: BandRow[] })
  | (CardBase & {
      kind: 'landcover';
      views: CardView[];
      /** A view's twin ranked within the area (hazards): shown by the Ukraine / This area toggle. */
      local: Record<string, CardView>;
      classes: ClassItem[];
      stats: Stat[];
      /** The period or condition the summary sentences share, said once above them. */
      lead?: string;
      summary: string[];
      transitions: TransitionItem[];
      /** Hazards, most prominent first; each selects its view (`key`). Empty for other land-cover cards. */
      hazards: HazardItem[];
      /** How far the result was checked, in one sentence. */
      checked?: string;
      tables: TableModel[];
    })
  | (CardBase & { kind: 'image'; views: [CardView] });

/** One hazard of a hazard card: its risk levels in this area, for the selector row. */
export interface HazardItem {
  key: string;  // the view it selects
  title: string;
  year: string;
  colour?: string;
  levels: { label: string; range: string; colour?: string; area: string; percent: number }[];
  high: string;  // "1.91 km² · 10 %"
  skill?: string;  // "AUC 0.98 vs 0.87 (best null)"
  sentence: string;  // "High risk: 21.4 km² of 169 km² of cropland and grassland"
  sparse: boolean;  // under 1 % of the area analysed could make this change
  highKm2: number;
}

/** One class-to-class flow of a land-cover card: "Trees → Crops · 0.79 km² · 4 % of Trees". */
export interface TransitionItem {
  from: string;
  to: string;
  fromColour?: string;
  toColour?: string;
  area: string;
  share: string;
}

/** A small table of a land-cover card: a label column (with an optional swatch), then values. */
export interface TableModel {
  title: string;
  columns: string[];
  rows: { label: string; colour?: string; cells: string[] }[];
  note?: string;
}

/** The files (or region) the analysis was asked about: the user message's context. */
export interface AnalysedContext {
  attachments: MessageAttachment[];
  region?: ChatRegion;
}

// --- Display tables ---------------------------------------------------------------

const INDEX_ARTIFACTS = ['NDVIMap', 'EVIMap', 'NDWIMap', 'NDSIMap', 'ChangeHazardMap'];  // ChangeHazardMap: old jobs

/** Esri land-cover classes, in the order of their raster values 1-8 (dta/dti/models/lulc_hazard/features.py). */
const ESRI_CLASSES = ['Water', 'Trees', 'FloodedVeg', 'Crops', 'Built', 'Bare', 'Snow', 'Rangeland'];

/** A land-cover result's classes in raster-value order (`metadata.legend`; Esri for jobs from before it existed) and the
 *  map's short name. */
interface LandLegend { names: string[]; label: string }

function landLegend(meta: Obj): LandLegend {
  const names = arr(meta.legend).map(str).filter(Boolean);
  if (names.length === 0) return { names: ESRI_CLASSES, label: 'Esri' };
  return { names, label: str(meta.source).startsWith('KPI') ? 'KPI-Ukraine' : 'Esri' };
}

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
 * that is not projected), "<0.1 km²" for a trace that would round to 0.0. `sign` puts
 * "+" before a positive value.
 */
export function formatKm2(value: unknown, { sign = false } = {}): string {
  if (!isNum(value)) return '—';
  const abs = Math.abs(value);
  if (abs > 0 && abs < 0.05) return '<0.1 km²';
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

/** The map-only PNG `<key>_bare` of the first visualization key that has one. */
const bareOf = (vis: Obj, ...keys: string[]): ArtifactRef | undefined =>
  keys.map((k) => asRef(vis[`${k}_bare`])).find(Boolean);

const isHex = (v: unknown): v is string => typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v);

/** A bare map's `stats.style`, or undefined when it has none (or not a valid one). */
export function styleOf(ref: ArtifactRef | undefined): RasterStyle | undefined {
  const s = obj(ref?.stats?.style);
  if ((s.kind === 'change' || s.kind === 'index') && Array.isArray(s.ramp) && s.ramp.length > 1 && s.ramp.every(isHex)) {
    const r = s.range;
    if (Array.isArray(r) && r.length === 2 && r.every(isNum)) return { kind: s.kind, range: [r[0], r[1]], ramp: s.ramp };
  }
  if (s.kind === 'class' && Object.values(obj(s.classes)).every(isHex)) return { kind: 'class', classes: obj(s.classes) as Record<string, string> };
  return undefined;
}

/** A change or index style as a ramp legend: its colour stops over its range. */
function rampOf(style: RasterStyle | undefined): { stops: string[]; range: [number, number] } | undefined {
  return style && style.kind !== 'class' ? { stops: style.ramp, range: style.range } : undefined;
}

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
function gpkgOf(artifact: Obj): CardBase['gpkg'] {
  const refs = [...Object.values(obj(artifact.rasters)), ...Object.values(obj(artifact.vectors)), artifact.gpkg].map(asRef);
  const ref = refs.find((r) => r && (r.kind === 'gpkg' || r.path.endsWith('.gpkg')));
  if (ref) return ref;
  const out = artifact.output_path;
  return typeof out === 'string' && out.toLowerCase().endsWith('.gpkg') ? { path: out, url: `/v1/download?path=${encodeURIComponent(out)}` } : undefined;
}

const titled = (...parts: string[]) => parts.filter(Boolean).join(' · ');

function indexCard(artifact: Obj, artifacts: Obj, context: AnalysedContext, bandMath: boolean): ResultCardModel {
  const index = String(artifact.index_type ?? '').toLowerCase();
  const name = bandMath ? String(artifact.name ?? artifact.formula ?? 'Band math') : index.toUpperCase();
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const file = byPath(context, artifacts.RasterPath) ?? byPath(context, artifact.path) ?? (files.length === 1 ? files[0] : undefined);
  const date = file?.metadata?.date;
  const stats = obj(artifact.statistics);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const image = asRef(vis[`${index}_map`]) ?? asRef(vis.index_map) ?? firstRef(vis, 'png');
  const bare = bareOf(vis, `${index}_map`, 'index_map');
  const ramp = rampOf(styleOf(bare));
  const raster = asRef(rasters[index]) ?? asRef(rasters.index) ?? firstRef(rasters, 'geotiff');
  const title = titled(name, dayMonth(date));
  return {
    kind: 'index',
    title,
    formula: bandMath && artifact.formula ? titled(prettyFormula(String(artifact.formula)), date ?? '', resolution(files)) : undefined,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    ramp,
    views: [{ key: 'map', label: 'Map', layerName: title, overlay: '', image, bare, raster, legend: ramp && { kind: 'ramp', ...ramp } }],
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

  const changeBare = bareOf(vis, 'change_map');
  const beforeBare = bareOf(vis, 'index_before', `${key}_before`);
  const afterBare = bareOf(vis, 'index_after', `${key}_after`);
  // Swatches from the change ramp: its first stop is lost, the middle no change, the last gained.
  const changeRamp = rampOf(styleOf(changeBare));
  const stops = changeRamp?.stops;
  const classes = changeClasses(artifact);
  const colour = (tone: Tone) => stops && (tone === 'lost' ? stops[0] : tone === 'gained' ? stops[stops.length - 1] : stops[(stops.length - 1) >> 1]);
  const item = (label: string, tone: Tone): LegendItem => ({ label, tone, ...(stops && { colour: colour(tone) }) });
  const legend: LegendItem[] = bandMath
    ? [item(lost, 'lost'), item(gained, 'gained'), item('no change', 'neutral')]
    : [item(gained, 'gained'), item(lost, 'lost'), item('no change', 'neutral')];
  const indexLegend = (bare: ArtifactRef | undefined): LayerLegend | undefined => {
    const ramp = rampOf(styleOf(bare));
    return ramp && { kind: 'ramp', ...ramp };
  };
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
        bare: changeBare,
        raster: asRef(rasters.change),
        legend: {
          kind: 'change',
          ...(changeRamp && { stops: changeRamp.stops, range: changeRamp.range }),
          ...(classes.length > 0 && { classes }),
        },
      },
      {
        key: 'before',
        label: 'Before',
        layerName: titled(shortName, before ?? 'before'),
        overlay: titled(`${shortName} before`, before ?? ''),
        image: asRef(vis.index_before) ?? asRef(vis[`${key}_before`]),
        bare: beforeBare,
        raster: asRef(rasters.index_before),
        legend: indexLegend(beforeBare),
      },
      {
        key: 'after',
        label: 'After',
        layerName: titled(shortName, after ?? 'after'),
        overlay: titled(`${shortName} after`, after ?? ''),
        image: asRef(vis.index_after) ?? asRef(vis[`${key}_after`]),
        bare: afterBare,
        raster: asRef(rasters.index_after),
        legend: indexLegend(afterBare),
      },
    ],
  };
}

const TONE_ORDER: Record<Tone, number> = { lost: 0, neutral: 1, gained: 2 };

/**
 * A change result's classes from `class_statistics` (`{name: {name, colour, area_km2, ...}}`):
 * "Water gained" → gained, "Drier" → lost, "No change" / "Stable" → neutral; lost first.
 * The label drops the subject ("gained"), as the map legend reads "lost 0.6 km²".
 */
function changeClasses(artifact: Obj): LegendClass[] {
  return Object.values(obj(artifact.class_statistics))
    .map(obj)
    .filter((c) => typeof c.name === 'string')
    .map((c): LegendClass => {
      const name = String(c.name);
      const tone: Tone = /gain|wetter|increase/i.test(name) ? 'gained' : /lost|loss|drier|decrease/i.test(name) ? 'lost' : 'neutral';
      return {
        label: name.toLowerCase().replace(/^(water|vegetation) /, ''),
        tone,
        ...(isHex(c.colour) && { colour: c.colour }),
        ...(isNum(c.area_km2) && { area_km2: c.area_km2 }),
      };
    })
    .sort((a, b) => TONE_ORDER[a.tone] - TONE_ORDER[b.tone]);
}

function singleFile(context: AnalysedContext, artifacts: Obj, artifact: Obj) {
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const file = byPath(context, artifacts.RasterPath) ?? (files.length === 1 ? files[0] : undefined);
  return { files, date: file?.metadata?.date, file };
}

/**
 * The class legend from `class_statistics` (`{name: {id, name, colour, pixels, percent,
 * area_km2}}`): classes with pixels, each with its area, or its share when the CRS
 * has no area. A class without `colour` (an older job) takes the bare map's style.
 */
function classItems(artifact: Obj, bare: ArtifactRef | undefined): ClassItem[] {
  const style = styleOf(bare);
  const styled = style?.kind === 'class' ? style.classes : {};
  return Object.entries(obj(artifact.class_statistics))
    .map(([key, v]): [string, Obj] => [key, obj(v)])
    .filter(([, c]) => isNum(c.pixels) && c.pixels > 0)
    .map(([key, c]) => ({
      name: typeof c.name === 'string' ? c.name : key,
      colour: isHex(c.colour) ? c.colour : styled[String(c.id ?? c.class_id)],
      value: isNum(c.area_km2) ? formatKm2(c.area_km2) : formatPercent(c.percent ?? c.percentage),
    }));
}

function lulcCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date } = singleFile(context, artifacts, artifact);
  const title = titled('Land cover', dayMonth(date));
  const vis = obj(artifact.visualizations);
  const bare = bareOf(vis, 'classification_map');
  return {
    kind: 'classification',
    title,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    classes: classItems(artifact, bare),
    views: [{
      key: 'map', label: 'Map', layerName: title, overlay: '',
      image: asRef(vis.classification_map),
      bare,
      raster: asRef(obj(artifact.rasters).classification),
    }],
  };
}

function snowCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date } = singleFile(context, artifacts, artifact);
  const title = titled('Snow and ice', dayMonth(date));
  const vis = obj(artifact.visualizations);
  const bare = bareOf(vis, 'snow_classification');
  const share = obj(artifact.statistics).snow_coverage_percent;
  // Before class_statistics (older jobs): the snow share alone, without colours.
  const classes = Object.keys(obj(artifact.class_statistics)).length > 0
    ? classItems(artifact, bare)
    : [
        { name: 'Snow / ice', value: formatPercent(share) },
        { name: 'Other', value: isNum(share) ? formatPercent(100 - share) : '—' },
      ];
  return {
    kind: 'classification',
    title,
    source: sourceLine({ ...context, attachments: files }),
    bounds: boundsOf(files),
    classes,
    views: [{
      key: 'map', label: 'Map', layerName: title, overlay: '',
      image: asRef(vis.snow_classification),
      bare,
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
    gpkg: gpkgOf(artifact),
    stats,
    views: [{ key: 'map', label: 'Map', layerName: title, overlay: '', image: asRef(obj(artifact.visualizations).field_boundaries) }],
  };
}

function statisticsCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const { files, date, file } = singleFile(context, artifacts, artifact);
  const codes = (file ?? files[0])?.metadata?.bands ?? [];
  // Keys are band codes (B2…B6) since BR3; `band_<n>` (older jobs) is the file's n-th band.
  const bands = Object.entries(obj(artifact.bands)).map(([key, v]) => {
    const n = Number(/^band_(\d+)$/.exec(key)?.[1]);
    const s = obj(v);
    return {
      band: Number.isFinite(n) ? (codes.find((b) => b.index === n)?.code ?? `Band ${n}`) : key,
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
// --- Land cover: hazards, forecast, history (dta/dti/models/lulc_hazard) -----------------

const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown): string => (typeof v === 'string' ? v : isNum(v) ? String(v) : '');

/**
 * A land-cover area in km² (the server's `*_km2`): two decimals under 10 km², one under 100,
 * whole and grouped above, "<0.01 km²" for a trace; a real minus and an optional "+".
 */
export function formatArea(value: unknown, { sign = false } = {}): string {
  if (!isNum(value)) return '—';
  const abs = Math.abs(value);
  const prefix = value < 0 ? MINUS : sign && value > 0 ? '+' : '';
  if (abs > 0 && abs < 0.005) return `${prefix}<0.01 km²`;
  const digits = abs < 9.995 ? abs.toFixed(2) : abs < 99.95 ? abs.toFixed(1) : grouped(abs);
  return `${prefix}${digits} km²`;
}

/** `<key>_km2`, or an older job's `<key>_ha` in km². */
const km2Of = (o: Obj, key: string): unknown => (isNum(o[`${key}_km2`]) ? o[`${key}_km2`] : isNum(o[`${key}_ha`]) ? (o[`${key}_ha`] as number) / 100 : undefined);

const signedPercent = (v: unknown) => (isNum(v) ? `${v < 0 ? MINUS : v > 0 ? '+' : ''}${Math.abs(v).toFixed(0)} %` : '—');

/** Colour of class `name` from a class style (keys are the raster values: position in `names` + 1). */
function classColour(style: RasterStyle | undefined, names: string[], name: string): string | undefined {
  return style?.kind === 'class' ? style.classes[String(names.indexOf(name) + 1)] : undefined;
}

function classView(vis: Obj, rasters: Obj, key: string, label: string, layerName: string, overlay: string, names: string[] = ESRI_CLASSES): CardView {
  const bare = bareOf(vis, `${key}_map`);
  const style = styleOf(bare);
  const classes: LegendClass[] =
    style?.kind === 'class'
      ? names.filter((_, i) => style.classes[String(i + 1)]).map((n) => ({ label: n, tone: 'neutral' as Tone, colour: classColour(style, names, n) }))
      : [];
  return {
    key, label, layerName, overlay,
    image: asRef(vis[`${key}_map`]), bare, raster: asRef(rasters[key]),
    ...(classes.length > 0 && { legend: { kind: 'change' as const, classes } }),
  };
}

function rankView(vis: Obj, rasters: Obj, key: string, label: string, layerName: string, overlay: string): CardView {
  const bare = bareOf(vis, `${key}_map`);
  const ramp = rampOf(styleOf(bare));
  return { key, label, layerName, overlay, image: asRef(vis[`${key}_map`]), bare, raster: asRef(rasters[key]), legend: ramp && { kind: 'ramp', ...ramp } };
}

/** The class legend of a class view, with each class's area from `rows` (class → hectares). */
function landClasses(view: CardView | undefined, areas: Record<string, unknown>, names: string[]): ClassItem[] {
  const style = styleOf(view?.bare);
  return names.filter((n) => isNum(areas[n]) && (areas[n] as number) > 0).map((n) => ({
    name: n,
    colour: classColour(style, names, n),
    value: formatArea(areas[n]),
  }));
}

/** Transition matrix `{classes, hectares, row_percent}` as a table: rows from, columns to, hectares (share of the row). */
function matrixTable(m: Obj, title: string, style: RasterStyle | undefined, names: string[], note?: string): TableModel | null {
  const classes = arr(m.classes).map(str);
  const area = (m.km2 ? arr(m.km2) : arr(m.hectares).map((r) => arr(r).map((v) => (isNum(v) ? v / 100 : v)))).map(arr);
  const pct = arr(m.row_percent).map(arr);
  if (classes.length === 0) return null;
  return {
    title,
    columns: ['From \\ to', ...classes],
    rows: classes.map((c, i) => ({
      label: c,
      colour: classColour(style, names, c),
      cells: classes.map((_, j) => (isNum(area[i]?.[j]) && (area[i][j] as number) > 0 ? `${formatArea(area[i][j])} (${Number(pct[i]?.[j] ?? 0).toFixed(0)} %)` : '·')),
    })),
    note,
  };
}

function classesTable(rows: unknown[], title: string, before: string, after: string, style: RasterStyle | undefined, names: string[]): TableModel {
  return {
    title,
    columns: ['Class', before, after, 'Change', 'Change %'],
    rows: rows.map(obj).map((r) => ({
      label: str(r.class),
      colour: classColour(style, names, str(r.class)),
      cells: [formatArea(km2Of(r, 'before')), formatArea(km2Of(r, 'after')), formatArea(km2Of(r, 'change'), { sign: true }), signedPercent(r.change_percent)],
    })),
  };
}

function transitionsOf(stats: Obj, style: RasterStyle | undefined, names: string[]): TransitionItem[] {
  return arr(stats.transitions).map(obj).map((t) => ({
    from: str(t.from),
    to: str(t.to),
    fromColour: classColour(style, names, str(t.from)),
    toColour: classColour(style, names, str(t.to)),
    area: formatArea(t.km2),
    share: `${formatPercent(t.percent_of_from)} of ${str(t.from)}`,
  }));
}

const largest = (t: TransitionItem[]): Stat => ({ label: 'Largest change', value: t[0] ? `${t[0].from} → ${t[0].to} ${t[0].area}` : '—' });

function hazardCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const stats = obj(artifact.statistics);
  const meta = obj(artifact.metadata);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const t0 = str(meta.t0);
  const legend = landLegend(meta);
  const start = classView(vis, rasters, 'start_class', `Land cover ${t0}`, titled('Land cover', t0), titled(`${legend.label} land cover`, t0), legend.names);
  // km² per class; an older job has hectares.
  const startAreas = stats.start_class_km2
    ? obj(stats.start_class_km2)
    : Object.fromEntries(Object.entries(obj(stats.start_class_ha)).map(([k, v]) => [k, isNum(v) ? v / 100 : v]));
  const highShare = (h: Obj) => (isNum(obj(arr(h.levels)[0]).percent) ? (obj(arr(h.levels)[0]).percent as number) : isNum(h.top_decile_percent) ? h.top_decile_percent : 0);
  // Named hazards by their share at high risk, then "any change" as the overview.
  const all = arr(stats.hazards).map(obj);
  const sorted = [...all.filter((h) => h.group !== 'any_change').sort((a, b) => highShare(b) - highShare(a)), ...all.filter((h) => h.group === 'any_change')];
  const views: CardView[] = [];
  const local: Record<string, CardView> = {};
  const hazards: HazardItem[] = [];
  const hazardEligible: number[] = [];
  for (const h of sorted) {
    const rank = str(h.raster);
    const title = str(h.title);
    const year = str(h.year);
    const levels = arr(h.levels).map(obj).map((l) => ({
      label: str(l.level),
      range: str(l.range),
      colour: isHex(l.colour) ? l.colour : undefined,
      area: formatArea(l.km2),
      percent: isNum(l.percent) ? l.percent : 0,
    }));
    const levelKey = str(h.level_raster);
    const main: CardView = levelKey
      ? {
          ...classView(vis, rasters, levelKey, title, `${title} · ${year}`, titled(`risk by ${year}`, 'vs Ukraine')),
          items: levels.map((l) => ({ name: `${l.label} · ${l.range}`, colour: l.colour, value: l.area })),
        }
      : rankView(vis, rasters, rank, title, `${title} · ${year}`, titled(`rank by ${year}`, 'vs Ukraine'));
    main.base = start.bare;
    views.push(main);
    local[main.key] = { ...rankView(vis, rasters, `${rank}_local`, title, `${title} · ${year} · in area`, titled(`rank by ${year}`, 'within this area')), base: start.bare };
    const s = obj(h.skill);
    const high = levels[0] ?? { area: formatArea(km2Of(h, 'top_decile')), percent: highShare(h) };
    const eligible = km2Of(h, 'eligible');
    const highKm2 = isNum(obj(arr(h.levels)[0]).km2) ? (obj(arr(h.levels)[0]).km2 as number) : isNum(km2Of(h, 'top_decile')) ? (km2Of(h, 'top_decile') as number) : 0;
    hazards.push({
      key: main.key,
      title,
      year,
      colour: isHex(h.colour) ? h.colour : undefined,
      levels,
      high: `${high.area} · ${formatPercent(high.percent)}`,
      skill: isNum(s.auc) ? `AUC ${Number(s.auc).toFixed(2)}${isNum(s.best_null_auc) ? ` vs ${Number(s.best_null_auc).toFixed(2)}` : ''}` : undefined,
      sentence: `High risk: ${high.area} of ${formatArea(eligible)} of ${str(h.land) || 'the land this applies to'}`,
      sparse: false,
      highKm2,
    });
    hazardEligible.push(isNum(eligible) ? eligible : 0);
  }
  views.push(start);
  const years = [...new Set(hazards.map((h) => h.year))];
  const longest = years.length ? String(Math.max(...years.map(Number))) : '—';
  const shorter = hazards.filter((h) => h.year !== longest).map((h) => `${h.title} ${h.year}`);
  const analysed_ = Object.values(startAreas).reduce<number>((a, v) => a + (isNum(v) ? v : 0), 0);
  hazards.forEach((h, i) => {
    if (hazardEligible[i] < 0.01 * analysed_) {
      h.sparse = true;
      h.sentence = `Little ${str(sorted[i].land) || 'land this applies to'} here (${formatArea(hazardEligible[i])})`;
    }
  });
  const named = hazards.filter((h) => !h.title.startsWith('Any') && !h.sparse);
  const top = named.length ? named.reduce((a, b) => (b.highKm2 > a.highKm2 ? b : a)) : undefined;
  return {
    kind: 'landcover',
    title: titled('Land-cover hazards', t0 && `from ${t0}`),
    note: 'risk levels are national ranks, not probabilities',
    source: `${legend.label} land cover, AlphaEarth and terrain from Earth Engine`,
    bounds: boundsOf(files),
    views,
    local,
    classes: landClasses(start, startAreas, legend.names),
    stats: [
      { label: 'Area analysed', value: formatArea(analysed_) },
      { label: 'Most at risk', value: top ? `${top.title} · ${top.levels[0]?.area ?? top.high} high` : '—' },
      { label: 'Horizon', value: shorter.length ? `${longest} (${shorter.join(', ')})` : longest },
    ],
    summary: [],
    transitions: [],
    hazards,
    checked: 'Each hazard is shown for the furthest year at which it beat simple rules in a 2021 backtest. High = among the 10 % most likely in Ukraine, compared with the same land.',
    tables: hazards.length
      ? [{
          title: 'Hazards',
          columns: ['Hazard', 'By', 'High', 'Elevated', 'Moderate', 'Backtest'],
          rows: hazards.map((h) => ({
            label: h.title,
            colour: h.colour,
            cells: [h.year, ...[0, 1, 2].map((i) => (h.levels[i] ? `${h.levels[i].area} (${formatPercent(h.levels[i].percent)})` : '—')), h.skill ?? '—'],
          })),
          note: 'Each cell is compared with the same land across Ukraine: high = among the 10 % most likely to make this change, elevated = the next 15 %, moderate = the next 25 %. These are ranks, not chances of change.',
        }]
      : [],
  };
}

function forecastCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const stats = obj(artifact.statistics);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const t0 = str(stats.t0);
  const year = str(stats.year);
  const legend = landLegend(obj(artifact.metadata));
  const names = legend.names;
  const now = classView(vis, rasters, 'class_start', String(t0 || 'Now'), titled('Land cover', t0), titled(`${legend.label} land cover`, t0), names);
  const changed = {
    ...classView(vis, rasters, 'changed', 'Changes', titled('Forecast changes', `${t0}–${year}`), titled('new class', `${t0} → ${year}`), names),
    base: now.bare,
  };
  const forecast = classView(vis, rasters, 'class_forecast', String(year || 'Forecast'), titled('Land-cover forecast', year), titled('forecast', year), names);
  const style = styleOf(forecast.bare);
  const rows = arr(stats.classes).map(obj);
  const trend = obj(stats.trend);
  const pair = arr(trend.pair).map(str);
  const transitions = transitionsOf(stats, style, names);
  return {
    kind: 'landcover',
    title: titled('Land-cover forecast', `${t0} → ${year}`),
    note: str(stats.amounts) === 'expected' ? 'expected amounts' : `scenario: ${pair.join('–')} trend`,
    source: `${legend.label} land cover, AlphaEarth and terrain from Earth Engine`,
    bounds: boundsOf(files),
    views: [changed, forecast, now],
    local: {},
    classes: landClasses(forecast, Object.fromEntries(rows.map((r) => [str(r.class), km2Of(r, 'after')])), names),
    stats: [
      { label: 'Changing', value: `${formatArea(stats.changed_km2)} · ${formatPercent(stats.changed_percent)}` },
      largest(transitions),
      { label: 'Horizon', value: `${str(stats.horizon_years) || '—'} years` },
    ],
    lead: str(stats.summary_lead) || undefined,
    summary: arr(stats.summary).map(str).filter(Boolean),
    transitions,
    hazards: [],
    checked: str(stats.checked) || undefined,
    tables: [
      classesTable(rows, 'Class areas', t0, year, style, names),
      matrixTable(obj(stats.matrix), `Transitions ${t0} → ${year}`, style, names),
      matrixTable(obj(trend.matrix), `Trend used: ${pair.join('–')} around the area`, style, names,
        `Observed ${legend.label} transitions in ${formatKm2(trend.area_km2)} around the area; they set how much changes.`),
    ].filter((t): t is TableModel => t !== null),
  };
}

function historyCard(artifact: Obj, artifacts: Obj, context: AnalysedContext): ResultCardModel {
  const files = analysed(context, [artifacts.RasterPath, artifact.path]);
  const stats = obj(artifact.statistics);
  const vis = obj(artifact.visualizations);
  const rasters = obj(artifact.rasters);
  const periods = obj(stats.periods);
  const span = (p: unknown) => arr(p).map(str).join('–');
  const first = span(periods.start);
  const last = span(periods.end);
  const legend = landLegend(obj(artifact.metadata));
  const names = legend.names;
  const start = classView(vis, rasters, 'class_start', first || 'Then', titled('Land cover', first), titled('majority class', first), names);
  const changed = {
    ...classView(vis, rasters, 'changed', 'Changes', titled('Land-cover changes', `${first} → ${last}`), titled('new class', `${first} → ${last}`), names),
    base: start.bare,
  };
  const end = classView(vis, rasters, 'class_end', last || 'Now', titled('Land cover', last), titled('majority class', last), names);
  const style = styleOf(end.bare);
  const years = arr(stats.years).map(str);
  const areas = obj(stats.area_km2 ?? stats.area_ha);
  const scale = stats.area_km2 ? 1 : 0.01;
  const rows = arr(stats.classes).map(obj);
  const transitions = transitionsOf(stats, style, names);
  return {
    kind: 'landcover',
    title: titled('Land-cover history', `${first} → ${last}`),
    note: `${legend.label}, majority of each period`,
    source: `${legend.label} land cover from Earth Engine`,
    bounds: boundsOf(files),
    views: [changed, end, start],
    local: {},
    classes: landClasses(end, Object.fromEntries(rows.map((r) => [str(r.class), km2Of(r, 'after')])), names),
    stats: [
      { label: 'Changed', value: `${formatArea(stats.changed_km2)} · ${formatPercent(stats.changed_percent)}` },
      largest(transitions),
      { label: `Single years ${arr(stats.single_year_pair).map(str).join('–')}`, value: formatPercent(stats.changed_percent_single_years) },
    ],
    lead: str(stats.summary_lead) || undefined,
    summary: arr(stats.summary).map(str).filter(Boolean),
    transitions,
    hazards: [],
    checked: str(stats.checked) || `Observed ${legend.label} labels; comparing 3-year majorities damps their year-to-year flicker.`,
    tables: [
      classesTable(rows, 'Class areas', first, last, style, names),
      matrixTable(obj(stats.matrix), `Transitions ${first} → ${last}`, style, names),
      {
        title: 'Area per year',
        columns: ['Class', ...years],
        rows: names.filter((n) => Array.isArray(areas[n])).map((n) => ({
          label: n,
          colour: classColour(style, names, n),
          cells: arr(areas[n]).map((v) => formatArea(isNum(v) ? v * scale : v)),
        })),
        note: `Single-year ${legend.label} maps: part of the year-to-year change is classifier flicker.`,
      },
      fallowTable(obj(stats.not_cultivated_km2)),
    ].filter((t): t is TableModel => t !== null),
  };
}

/** Area the KPI map marks as not cultivated (raw class 20) per year, `{years, km2}`; null when absent. */
function fallowTable(f: Obj): TableModel | null {
  const years = arr(f.years).map(str);
  if (years.length === 0) return null;
  return {
    title: 'Marked not cultivated (KPI class 20)',
    columns: ['', ...years],
    rows: [{ label: 'Area', cells: arr(f.km2).map((v) => formatArea(v)) }],
    note: 'Counted as Crop in the class tables above; abandoned cropland shows here, not as a change to Grass.',
  };
}

function imageCard(artifact: Obj, context: AnalysedContext, title: string): ResultCardModel | null {
  const image = firstRef(artifact.visualizations, 'png');
  if (!image) return null;
  return {
    kind: 'image',
    title,
    source: sourceLine(context),
    bounds: boundsOf(context.attachments),
    gpkg: gpkgOf(artifact),
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
  if (has('LandCoverForecast')) return forecastCard(get('LandCoverForecast'), artifacts, context);
  if (has('LandCoverHistory')) return historyCard(get('LandCoverHistory'), artifacts, context);
  if (has('ChangeHazardMap') && Array.isArray(obj(get('ChangeHazardMap').statistics).hazards)) {
    return hazardCard(get('ChangeHazardMap'), artifacts, context);
  }
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

/** A card view's raster as a map layer from the chat (`result-<jobId>-<view>`): On map, and the Library's Map. */
export function resultLayerOf(jobId: string, model: ResultCardModel, view: CardView & { raster: ArtifactRef }): Layer {
  return {
    id: `result-${jobId}-${view.key}`,
    name: view.layerName,
    type: 'raster',
    visible: true,
    opacity: 0.8,
    url: view.raster.path,
    bounds: model.bounds,
    legend: view.legend,
    source: 'chat',
    subtitle: 'from chat',
    addedAt: new Date().toISOString(),
  };
}
